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

type NeuralNetworkBot struct {
	net *network.Network
}

func (r NeuralNetworkBot) Decision(state GameState) int {
	_ = r.net.LoadSensors([]float64{
		float64(state.aPrevious),
		float64(state.bPrevious),
	})

	_, _ = r.net.Activate()
	outputs := r.net.ReadOutputs()

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
