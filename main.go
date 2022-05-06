package main

import (
	"context"
	"fmt"
	"github.com/yaricom/goNEAT/v2/experiment"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"golang.org/x/exp/rand"
	"log"
	"os"
	"time"
)

func main() {
	seed := time.Now().Unix()
	rand.Seed(uint64(seed))

	// Load neatOptions configuration
	configFile, err := os.Open("./xor.neat")
	if err != nil {
		log.Fatal("Failed to open context configuration file: ", err)
	}
	options, err := neat.LoadNeatOptions(configFile)
	if err != nil {
		log.Fatal("Failed to load NEAT options: ", err)
	}

	exp := experiment.Experiment{
		Id:       0,
		Trials:   make(experiment.Trials, 100),
		RandSeed: seed,
	}

	exp.MaxFitnessScore = 16

	var evaluator PrisonersDilemmaGenerationEvaluator
	// This special constructor creates a Genome with in inputs, out outputs, n out of maxHidden hidden units, and random
	// connectivity.  If rec is true then recurrent connections will be included. The last input is a bias
	// link_prob is the probability of a link. The created genome is not modular.
	// newId, in, out, n, maxHidden int, recurrent bool, linkProb float64
	genomeRand := genetics.NewGenomeRand(0, 2, 1, 1, 10, false, 0.7)

	// seems to make no difference for the thing I am trying to resolve between the above and below

	//genomeFile, err := os.Open("./xorstartgenes")
	//if err != nil {
	//	log.Fatal("Failed to open genome file: ", err)
	//}
	//startGenome, err := genetics.ReadGenome(genomeFile, 1)
	//if err != nil {
	//	log.Fatal("Failed to read start genome: ", err)
	//}
	//fmt.Println(startGenome)

	ctx, _ := context.WithCancel(context.Background())
	err = exp.Execute(neat.NewContext(ctx, options), genomeRand, evaluator, nil)
	if err != nil {
		fmt.Println(err.Error())
	}

	exp.PrintStatistics()
}

type PrisonersDilemmaGenerationEvaluator struct{}

func (ex PrisonersDilemmaGenerationEvaluator) GenerationEvaluate(
	pop *genetics.Population,
	epoch *experiment.Generation,
	context *neat.Options,
) (err error) {
	// Calculate the fitness of all organisms in the population
	// going to fight against RandomBot
	for _, org := range pop.Organisms {
		res, err := ex.orgEvaluate(org)
		if err != nil {
			return err
		}

		if res && (epoch.Best == nil || org.Fitness > epoch.Best.Fitness) {
			epoch.Solved = true
			epoch.WinnerNodes = len(org.Genotype.Nodes)
			epoch.WinnerGenes = org.Genotype.Extrons()
			epoch.WinnerEvals = context.PopSize*epoch.Id + org.Genotype.Id
			epoch.Best = org
			if epoch.WinnerNodes == 5 {
				neat.InfoLog(fmt.Sprintf("Dumped optimal genome\n"))
			}
		}
	}

	epoch.FillPopulationStatistics(pop)

	return nil
}

func (e *PrisonersDilemmaGenerationEvaluator) orgEvaluate(organism *genetics.Organism) (bool, error) {
	game := CreateGame()

	b := RandomBot{}
	netDepth, _ := organism.Phenotype.MaxActivationDepthFast(0) // The max depth of the network to be activated
	net := organism.Phenotype

	for !game.GameOver() {
		// get the game state
		state := game.State()

		// set up our input
		_ = net.LoadSensors([]float64{
			float64(state.aPrevious),
			float64(state.bPrevious),
		})

		// run the network
		_, _ = net.ForwardSteps(netDepth)

		decision := Cooperate
		if organism.Phenotype.Outputs[0].Activation > 0.5 {
			decision = Defect
		}

		game.Play(gameDecision{
			aChoice: decision,
			bChoice: b.Decision(state),
		})
	}

	organism.Fitness = float64(game.AScore)
	organism.Error = 0.0
	organism.IsWinner = rand.Intn(10)%2 == 0

	return false, nil
}

//
//func (ex PrisonersDilemmaGenerationEvaluator) GenerationEvaluate(
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

//https://github.com/yaricom/goNEAT/blob/master/executor.go
//https://maori.geek.nz/learning-to-play-asteroids-in-golang-with-neat-f44c3472938f
func runGames() {
	rand.Seed(uint64(time.Now().UnixNano()))

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
}
