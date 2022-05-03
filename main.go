package main

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/experiment"
	"golang.org/x/exp/rand"
	"time"
)

func main() {
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
		"OftenRandomDefectBot": OftenRandomDefectBot{}gi,
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

		//fmt.Println(k1, "win", k1Wins)/*
		//fmt.Println(k1, "draw", k1Draws)
		//fmt.Println(k1, "loss", k1Loses)*/
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
