package main

const (
	Cooperate = iota
	Defect
)

type Game struct {
	AScore    int
	BScore    int
	Round     int
	APrevious int
	BPrevious int
}

func CreateGame() Game {
	return Game{
		AScore:    0,
		BScore:    0,
		Round:     0,
		APrevious: 0,
		BPrevious: 0,
	}
}

type GameState struct {
	aPrevious int
	bPrevious int
	round     int
}

type gameDecision struct {
	aChoice int
	bChoice int
}

func (g *Game) State() GameState {
	return GameState{
		aPrevious: g.APrevious,
		bPrevious: g.BPrevious,
		round:     g.Round,
	}
}

func (g *Game) GameOver() bool {
	if g.Round > 10 {
		return true
	}

	return false
}

func (g *Game) Play(d gameDecision) {
	// if both play nice then both get a small reward
	if d.aChoice == Cooperate && d.bChoice == Cooperate {
		g.AScore += 1
		g.BScore += 1
	}

	// if both defect then nothing
	if d.aChoice == Defect && d.bChoice == Defect {
		g.AScore--
		g.BScore--
	}

	// if you cooperate and they don't you get a punishment
	// and they get a reward
	if d.aChoice == Cooperate && d.bChoice == Defect {
		g.BScore += 3
		g.AScore -= 2
	}
	if d.aChoice == Defect && d.bChoice == Cooperate {
		g.AScore += 3
		g.BScore -= 2
	}

	// keep what happened last round so we can feed that back
	g.APrevious = d.aChoice
	g.BPrevious = d.bChoice

	// increment the round
	g.Round++
}
