package main

import (
	"github.com/yaricom/goNEAT/v2/neat/genetics"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"golang.org/x/exp/rand"
	"strings"
)

type Bot interface {
	Decision(state GameState) int
}

type RandomBot struct{}

func (r RandomBot) Decision(state GameState) int {
	return rand.Intn(2)
}

type DefectBot struct{}

func (r DefectBot) Decision(state GameState) int {
	return Defect
}

type CooperateBot struct{}

func (r CooperateBot) Decision(state GameState) int {
	return Cooperate
}

type TitForTatBot struct{}

func (r TitForTatBot) Decision(state GameState) int {
	if state.aPrevious == Defect {
		return Defect
	}
	return Cooperate
}

type TitForTatBotReverse struct{}

func (r TitForTatBotReverse) Decision(state GameState) int {
	if state.aPrevious == Cooperate {
		return Defect
	}
	return Cooperate
}

type RandomDefectBot struct{}

func (r RandomDefectBot) Decision(state GameState) int {
	if rand.Intn(10) == 0 {
		return Defect
	}
	return Cooperate
}

type OftenRandomDefectBot struct{}

func (r OftenRandomDefectBot) Decision(state GameState) int {
	if rand.Intn(3) == 0 {
		return Defect
	}
	return Cooperate
}

type NeuralNetworkBot struct{}

func (r NeuralNetworkBot) Decision(state GameState) int {
	net := getGenome(`/* Organism #87 Fitness: 25.000 Error: 0.000 */
genomestart 87
trait 1 0 0 0 0 0 0 0 0
node 1 1 1 1 SigmoidSteepenedActivation
node 2 1 1 3 SigmoidSteepenedActivation
node 3 1 0 0 SigmoidSteepenedActivation
node 13 1 0 2 SigmoidSteepenedActivation
gene 1 2 3 -0.3008783999474857 false 27 -0.3008783999474857 true
gene 1 2 13 0.5708516273454957 false 157 0.5708516273454957 true
gene 1 3 13 0.6841751300974551 false 158 0.6841751300974551 true
genomeend 87
`)

	_ = net.LoadSensors([]float64{
		float64(state.aPrevious),
		float64(state.bPrevious),
	})

	_, _ = net.Activate()
	outputs := net.ReadOutputs()

	// based on what the network says play!
	decision := Cooperate
	if outputs[0] > 0.5 {
		decision = Defect
	}

	return decision
}

func getGenome(genomeStr string) *network.Network {
	genome, _ := genetics.ReadGenome(strings.NewReader(genomeStr), 1)

	net, _ := genome.Genesis(1)

	return net
}
