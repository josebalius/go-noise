package noise

type Grad struct {
	X, Y, Z float64
}

func NewGrad(x, y, z float64) *Grad {
	return &Grad{x, y, z}
}

func (g *Grad) Dot2(x, y float64) float64 {
	return g.X*x + g.Y*y
}

func (g *Grad) Dot3(x, y, z float64) float64 {
	return g.X*x + g.Y*y + g.Z*z
}
