package main

import (
	"context"
	"fmt"
	"github.com/yaricom/goNEAT/v2/experiment"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"golang.org/x/exp/rand"
	"time"
)

func main() {
	// create experiment
	exp := experiment.Experiment{
		Id:       0,
		Trials:   make(experiment.Trials, 100),
		RandSeed: time.Now().UnixMilli(),
	}
	options := neat.Options{
		LogLevel:          "info",
		NumRuns:           100,
		PopSize:           100,
		CompatThreshold:   1,
		EpochExecutorType: neat.EpochExecutorTypeParallel,
	}
	exp.MaxFitnessScore = 16

	// This special constructor creates a Genome with in inputs, out outputs, n out of maxHidden hidden units, and random
	// connectivity.  If rec is true then recurrent connections will be included. The last input is a bias
	// link_prob is the probability of a link. The created genome is not modular.
	// newId, in, out, n, maxHidden int, recurrent bool, linkProb float64
	genomeRand := genetics.NewGenomeRand(0, 2, 1, 1, 10, false, 0.7)

	err := exp.Execute(neat.NewContext(context.TODO(), &options), genomeRand, AsteroidGenerationEvaluator{}, nil)
	if err != nil {
		fmt.Println(err.Error())
	}
	//runGames()
}

type AsteroidGenerationEvaluator struct{}

//GenerationEvaluate(pop *genetics.Population, epoch *Generation, context *neat.Options) (err error)
func (ex AsteroidGenerationEvaluator) GenerationEvaluate(
	pop *genetics.Population,
	epoch *experiment.Generation,
	context *neat.Options,
) (err error) {
	// Calculate the fitness of all organisms in the population
	fmt.Println("here")
	for _, org := range pop.Organisms {
		fmt.Println(org)
	}

	return nil
}

//https://github.com/yaricom/goNEAT/blob/master/executor.go
//https://maori.geek.nz/learning-to-play-asteroids-in-golang-with-neat-f44c3472938f
func runGames() {
	rand.Seed(uint64(time.Now().UnixNano()))
	e := experiment.Experiment{}

	fmt.Println(e)
	//e.Execute(context.TODO(), )
	//err := e.Execute(
	//	context.TODO(),
	//	start_genome,
	//	AsteroidGenerationEvaluator{
	//		OutputPath:         out_dir_path,
	//		PlayTimeInSeconds:  120,
	//		FrameRatePerSecond: 15,
	//	}
	//)

	// create the bots and play them against each other and print how they did over 1000 games
	bots := map[string]Bot{
		"RandomBot":            RandomBot{},
		"TitForTatBot":         TitForTatBot{},
		"DefectBot":            DefectBot{},
		"CooperateBot":         CooperateBot{},
		"RandomDefectBot":      RandomDefectBot{},
		"TitForTatBotReverse":  TitForTatBotReverse{},
		"OftenRandomDefectBot": OftenRandomDefectBot{},
	}

	winRates := map[string]float64{}
	lossRates := map[string]float64{}
	drawRates := map[string]float64{}

	scoreRates := map[string]int{}

	for k1, b1 := range bots {
		// now run X times and see how they go
		k1Wins := 0
		k1Loses := 0
		k1Draws := 0

		gameTurns := 100_000
		for _, b2 := range bots {
			for i := 0; i < gameTurns; i++ {
				game := CreateGame()

				game.Play(gameDecision{
					aChoice: -1,
					bChoice: -1,
				})

				for !game.GameOver() {
					state := game.State()
					game.Play(gameDecision{
						aChoice: b1.Decision(state),
						bChoice: b2.Decision(state),
					})
				}

				if game.AScore == game.BScore {
					k1Draws++
				}
				if game.AScore > game.BScore {
					k1Wins++
				}
				if game.AScore < game.BScore {
					k1Loses++
				}

				scoreRates[k1] += game.AScore
			}
		}

		winRates[k1] = (float64(k1Wins) / float64(gameTurns*len(bots))) * 100
		lossRates[k1] = (float64(k1Loses) / float64(gameTurns*len(bots))) * 100
		drawRates[k1] = (float64(k1Draws) / float64(gameTurns*len(bots))) * 100
	}

	for k, v := range winRates {
		fmt.Println()
		fmt.Println(k, "winRate", v)
		fmt.Println(k, "lossRate", lossRates[k])
		fmt.Println(k, "drawRate", drawRates[k])

		fmt.Println(k, "win+DrawRate", v+drawRates[k])
	}

	fmt.Println("")
	for k, v := range scoreRates {
		fmt.Println(k, "score", v)
	}
	//
	//game := CreateGame()
	//rBot := RandomBot{}
	//tBot := TitForTatBot{}
	//
	//game.Play(gameDecision{
	//	aChoice: -1,
	//	bChoice: -1,
	//})
	//
	//for !game.GameOver() {
	//	state := game.State()
	//	game.Play(gameDecision{
	//		aChoice: rBot.Decision(state),
	//		bChoice: tBot.Decision(state),
	//	})
	//}
	//
	//fmt.Println(game.Round)
	//fmt.Println(game.AScore)
	//fmt.Println(game.BScore)
}

func playGame() {
	// create a game
	// get the inputs
	// pass inputs to network
	// run the network
	// read the output
	// update the state
	// if game over quit
}

//
//func (ex AsteroidGenerationEvaluator) GenerationEvaluate(
//	population *genetics.Population,
//	epoch *experiments.Generation,
//	context *neat.NeatContext,
//) (err error) {
//	// Calculate the fitness of all organisms in the population
//	for _, org := range population.Organisms {
//		net := org.Phenotype // Neural Network
//		game := &asteroids.Game{}
//		frames := PlayTimeInSeconds * FrameRatePerSecond
//		for f = 0; f < frames; f++ {
//			// calculate the inputs
//			inputs := FindInputs(game)
//			// send those inputs to the network
//			net.LoadSensors(inputs)
//			net.Activate() // run the network
//			for i, output := range net.ReadOutputs() {
//				// if a key output is pushed
//				switch i {
//				case 0:
//					key = keys.KEY_UP
//				case 1:
//					key = keys.KEY_SPACE
//				case 2:
//					key = keys.KEY_LEFT
//				case 3:
//					key = keys.KEY_RIGHT
//				}
//				if output > 0.5 { // output activated
//					pressedKeys[key] = true
//				}
//			}
//			// Update the GameState
//			game.Update(pressedKeys)
//			if game.GameOver() {
//				break
//			}
//		}
//		// Use game score as fitness function
//		// Fitness is normalized to between 0 and 1
//		org.Fitness = norm(game.Score)
//	}
//}