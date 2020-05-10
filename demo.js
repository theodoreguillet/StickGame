async function input(question) {
    term.write(question);
    return new Promise((resolve, reject) => {
        term.on('key', function(key) {
            resolve(key);
        });
    });
}

function print(text) {
    if(!text) {
        text = "";
    }
    term.write(text + "\n\r");
}

/**
 * Choisit un élément aléatoire dans un tableau
 * @param {Array} values 
 */
function randChoice(values) {
    return values[Math.floor(Math.random()*values.length)];
}

class Player {
    constructor() {
        /**
         * @type {Array<{prevState: Number; action: Number; nextState: Number; reward: Number}>}
         */
        this.history = [];
        this.rewards = [];
        this.winCount = 0;
        this.loseCount = 0;
    }

    resetStats() {
        this.winCount = 0;
        this.loseCount = 0;
        this.rewards = [];
    }

    gameRestart() {
        this.history = [];
    }

    /**
     * Play from current state, return next state
     * @param {Number} state 
     * @param {Array<Number>} possibleNextStates
     */
    async play(state, possibleNextStates) {
        return possibleNextStates[0];
    }

    /**
     * On game turn finished
     * @param {Boolean} isPlayerTurn 
     * @param {Boolean} isGameOver 
     * @param {Number} prevState 
     * @param {Number} nextState 
     * @param {Number} action 
     */
    turnFinished(isPlayerTurn, isGameOver, prevState, nextState, action) {
        if(isPlayerTurn) {
            // Add transition to history
            this.history.push({
                action,
                prevState,
                nextState: NaN,
                reward: 0
            });
            if(isGameOver) {
                // Player lose
                this.history[this.history.length - 1].reward = -1;
                this.rewards.push(-1);
                this.loseCount++;
            }
        } else if(this.history.length > 0) {
            // Update last transition with new state and reward
            this.history[this.history.length - 1].nextState = nextState;
            if(isGameOver) {
                // Player win
                this.history[this.history.length - 1].reward = 1;
                this.rewards.push(1);
                this.winCount++;
            }
        }
    }
}

class PlayerAI extends Player {
    /**
     * Trainable player with ai
     * @param {Object} values load trained agent
     * @param {Number} eps epsilon
     * @param {Number} lr learning rate
     */
    constructor(values = {}, eps = 0.99, lr = 0.001) {
        super();
        this.eps = eps;
        this.lr = lr;
        this.values = values;
    }

    async play(state, possibleNextStates) {
        const r = Math.random();
        // espilon-greedy
        if(r < this.eps) {
            // exploration
            return randChoice(possibleNextStates);
        } else {
            return this.greedyStep(state, possibleNextStates);
        }
    }

    greedyStep(currentState, possibleNextStates) {
        // Return the state with the minimum value, we choose the worst state for the opponent
        const worstNextState = possibleNextStates.reduce((min, current) => 
            (this.values[current] < this.values[min]) ? current : min
        );
        return worstNextState;
    }

    train() {
        for(const action of this.history.reverse()) {
            let prevStateValue = this.values[action.prevState] || 0;
            let nextStateValue = this.values[action.nextState] || 0;

            if(action.reward === 0) {
                prevStateValue += this.lr * (nextStateValue - prevStateValue);
            } else {
                prevStateValue += this.lr * (action.reward - prevStateValue);
            }

            this.values[action.prevState] = prevStateValue;
        }
    }
}

class RandomPlayer extends Player {
    async play(state, possibleNextStates) {
        return randChoice(possibleNextStates);
    }
}

class HumanPlayer extends Player {
    async play(state, possibleNextStates) {
        while(true) {
            print();
            const userInput = await input("Take 1, 2 or 3 sticks ?: ");
            const action = parseInt(userInput, 10);
            if(action !== NaN && action > 0 && action <= possibleNextStates.length) {
                return possibleNextStates[action - 1];
            } else {
                print();
                print(`Bad number: ${action}`)
            }
        }
    }
}

class StickGame {
    /**
     * Stick game
     * @param {Player} player1 
     * @param {Player} player2 
     * @param{0|1} startPlayer
     * @param {Number} nstick 
     * @param {Array<Number>} actions 
     */
    constructor(player1, player2, startPlayer = 0, size = 24, actions = [ 1, 2, 3 ]) {
        this.player1 = player1;
        this.player2 = player2;
        this.size = size;
        this.actions = actions;

        this.restart(startPlayer);
    }

    restart(startPlayer = 0) {
        this.nstick = this.size;
        this.finished = false;
        this.player = !!startPlayer;

        this.player1.gameRestart();
        this.player2.gameRestart();
    }

    display() {
        let gamePrint = "";
        for(let i = 0; i < this.nstick; i++) {
            gamePrint += "| ";
        }
        print(gamePrint);
    }

    isFinished() {
        return this.finished;
    }

    getPlayer(opponent = false) {
        if(
            (this.player === false && !opponent) ||
            (this.player === true && opponent)
        ) {
            return this.player1;
        } else {
            return this.player2;
        }
    }

    async playNextTurn() {
        const player = this.getPlayer(false);
        const opponent = this.getPlayer(true);

        const possibleNStick = this.actions.map(action => this.nstick - action);

        const prevNStick = this.nstick;
        this.nstick = await player.play(this.nstick, possibleNStick);
        
        if(this.nstick <= 0) {
            this.finished = true;
        }

        player.turnFinished(true, this.finished, prevNStick, this.nstick, prevNStick - this.nstick);
        opponent.turnFinished(false, this.finished, prevNStick, this.nstick, prevNStick - this.nstick);

        this.player = !this.player;
    }
}

/**
 * 
 * @param {PlayerAI} playerAI 
 * @param {PlayerAI} opponentAI 
 */
async function train(playerAI, opponentAI, nepoch = 10000) {
    const trainGame = new StickGame(playerAI, opponentAI, randChoice([ 0, 1 ]));

    for(let epoch = 0; epoch < nepoch; epoch++) {
        while(!trainGame.isFinished()) {
            await trainGame.playNextTurn();
        }

        playerAI.train();
        opponentAI.train();

        playerAI.eps = Math.max(playerAI.eps * 0.996, 0.05);
        opponentAI.eps = Math.max(opponentAI.eps * 0.996, 0.05);

        trainGame.restart(randChoice([ 0, 1 ]));
    }
}

async function test(playerAI, ngames = 1000) {
    const testGame = new StickGame(playerAI, new RandomPlayer());
    
    for(let i = 0; i < ngames; i++) {
        testGame.restart(randChoice([ 0, 1 ]));
        
        while(!testGame.isFinished()) {
            await testGame.playNextTurn();
        }
    }
}

async function main() {
    const playerAI = new PlayerAI();
    const opponentAI = new PlayerAI();

    await train(playerAI, opponentAI);
    playerAI.resetStats();

    await test(playerAI);

    print("Values", playerAI.values);

    print(`Win rate: ${playerAI.winCount / (playerAI.winCount + playerAI.loseCount)}`);

    print(`Win mean: ${playerAI.rewards.reduce((sum, r) => sum + r, 0) / playerAI.rewards.length}`);

    const player = new HumanPlayer();
    const game = new StickGame(player, playerAI);

    playerAI.eps = 0;

    while(true) {
        playerAI.resetStats();
        game.restart(0);

        let i = 0;
        while(!game.isFinished()) {
            if(i > 0 && i % 2 === 0) {
                print("AI played, result:");
            }
            game.display();
            await game.playNextTurn();
            print();
            i++;
        }
        if(playerAI.winCount > 0) {
            print("Game Over...");
        } else {
            print("You win !");
        }
        print("=======================");
        print();
    }
}

main();
