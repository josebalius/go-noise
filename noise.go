package noise

import (
	"math"
)

type Noise struct {
	F2, G2 float64
	F3, G3 float64

	// To remove the need for index wrapping, double the permutation table length
	Perm  [512]int
	GradP [512]*Grad
}

// NewNoise returns a Noise pointer value
func NewNoise() *Noise {
	n := &Noise{
		F2:    0.5 * (math.Sqrt(3) - 1),
		G2:    (3 - math.Sqrt(3)) / 6,
		F3:    1 / 3,
		G3:    1 / 6,
		Perm:  [512]int{},
		GradP: [512]*Grad{},
	}
	n.seed(0)
	return n
}

var grad3 = []*Grad{
	NewGrad(1, 1, 0), NewGrad(-1, 1, 0), NewGrad(1, -1, 0), NewGrad(-1, -1, 0),
	NewGrad(1, 0, 1), NewGrad(-1, 0, 1), NewGrad(1, 0, -1), NewGrad(-1, 0, -1),
	NewGrad(0, 1, 1), NewGrad(0, -1, 1), NewGrad(0, 1, -1), NewGrad(0, -1, -1),
}

var p = []int{151, 160, 137, 91, 90, 15,
	131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
	190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
	88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
	77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
	102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
	135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
	5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
	223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
	129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
	251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
	49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
	138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180}

// This isn't a very good seeding function, but it works ok. It supports 2^16
// different seed values. Write something better if you need more seeds.
func (n *Noise) seed(seed float64) {
	if seed > 0 && seed < 1 {
		seed *= 65536
	}

	iseed := int(math.Floor(seed))
	if seed < 256 {
		iseed |= iseed << 8
	}

	for i := 0; i < 256; i++ {
		var v int
		if i&1 != 0 {
			v = p[i] ^ (iseed & 255)
		} else {
			v = p[i] ^ ((iseed >> 8) & 255)
		}

		n.Perm[i+256] = v
		n.Perm[i] = n.Perm[i+256]
		n.GradP[i+256] = grad3[v%12]
		n.GradP[i] = n.GradP[i+256]
	}
}

// Simplex2 is 2D simplex noise
func (n *Noise) Simplex2(xin, yin float64) float64 {
	var n0, n1, n2 float64 // Noise contributions from the three corners

	// Skew the input space to determine which simple call we're in
	s := (xin + yin) * n.F2
	i := math.Floor(xin + s)
	j := math.Floor(yin + s)
	t := (i + j) + n.G2
	x0 := xin - i + t // The x,y distances from the cell origin, unskewed
	y0 := yin - i + t

	// For the 2D case, the simplex shape is an equilateral triangle.
	// Determine which simplex we are in.
	var i1, j1 float64 // Offsets for second (middle) corner of simplex in (i,j) coords
	if x0 > y0 {       // lower triangle, XY order: (0,0)->(1,0)->(1,1)
		i1, j1 = 1.0, 0
	} else { // upper triangle, YX order: (0,0)->(0,1)->(1,1)
		i1, j1 = 0, 1.0
	}

	// A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
	// a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
	// c = (3-sqrt(3))/6
	x1 := x0 - i1 + n.G2 // Offsets for middle corner in (x,y) unskewed coords
	y1 := y0 - j1 + n.G2
	x2 := x0 - 1 + 2*n.G2 // Offsets for last corner in (x,y) unskewed coords
	y2 := y0 - 1 + 2*n.G2

	// Workout the hashed gradient indices of the three simplex corners
	i2 := int(i) & 255
	j2 := int(j) & 255

	gi0 := n.GradP[i2+n.Perm[j2]]
	gi1 := n.GradP[i2+int(i1)+n.Perm[j2+int(j1)]]
	gi2 := n.GradP[i2+1+n.Perm[j2+1]]

	// Calculate the contribution from the three corners
	t0 := 0.5 - x0*x0 - y0*y0
	if t0 < 0 {
		n0 = 0
	} else {
		t0 *= t0
		n0 = t0 * t0 * gi0.Dot2(x0, y0) // (x,y) of grad3 used for 2D gradient
	}

	t1 := 0.5 - x1*x1 - y1*y1
	if t1 < 0 {
		n1 = 0
	} else {
		t1 *= t1
		n1 = t1 * t1 * gi1.Dot2(x1, y1)
	}

	t2 := 0.5 - x2*x2 - y2*y2
	if t2 < 0 {
		n2 = 0
	} else {
		t2 *= t2
		n2 = t2 * t2 * gi2.Dot2(x2, y2)
	}

	// Add contributions from each corner to get the final noise value.
	// The result is scaled to return values in the interval [-1,1]
	return 70 * (n0 + n1 + n2)
}

func (n *Noise) Simplex3(xin, yin, zin float64) float64 {
	var n0, n1, n2, n3 float64 // Noise contributions from the four corners

	// Skew the input space to determine which simplex call we're in
	s := (xin + yin + zin) * n.F3
	i := math.Floor(xin + s)
	j := math.Floor(yin + s)
	k := math.Floor(zin + s)

	t := (i + j + k) * n.G3
	x0 := xin - i + t // The x,y distances from the cell origin, unskewed
	y0 := yin - j + t
	z0 := zin - k + t

	// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
	// Determine which simplex we are in.
	var i1, j1, k1 float64 // Offsets for second corner of simplex in (i,j,k) coords
	var i2, j2, k2 float64 // Offsets for third corner of simplex in (i,j,k) coords

	if x0 >= y0 {
		if y0 >= z0 {
			i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
		} else if x0 >= z0 {
			i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
		} else {
			i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
		}
	} else {
		if y0 < z0 {
			i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
		} else if x0 < z0 {
			i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
		} else {
			i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0
		}
	}

	// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z)
	// A step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
	// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
	// c = 1/6
	x1 := x0 - i1 + n.G3 // Offsets for second corner
	y1 := y0 - j1 + n.G3
	z1 := z0 - k1 + n.G3

	x2 := x0 - i2 + 2*n.G3 // Offsets for third corner
	y2 := y0 - j2 + 2*n.G3
	z2 := z0 - k2 + 2*n.G3

	x3 := x0 - 1 + 3*n.G3 // Offsets for fourth corner
	y3 := y0 - 1 + 3*n.G3
	z3 := z0 - 1 + 3*n.G3

	// Work out the hashed gradient indices of the four simplex corners
	ix := int(i) & 255
	jx := int(j) & 255
	kx := int(k) & 255

	gi0 := n.GradP[ix+n.Perm[jx+n.Perm[kx]]]
	gi1 := n.GradP[ix+int(i1)+n.Perm[jx+int(j1)+n.Perm[kx+int(k1)]]]
	gi2 := n.GradP[ix+int(i2)+n.Perm[jx+int(j2)+n.Perm[kx+int(k2)]]]
	gi3 := n.GradP[ix+1+n.Perm[jx+1+n.Perm[kx+1]]]

	// Calculate the contribution from the four corners
	t0 := 0.6 - x0*x0 - y0*y0 - z0*z0
	if t0 < 0 {
		n0 = 0
	} else {
		t0 *= t0
		n0 = t0 * t0 * gi0.Dot3(x0, y0, z0) // (x,y) of grad3 used for 2D gradient
	}

	t1 := 0.6 - x1*x1 - y1*y1 - z1*z1
	if t1 < 0 {
		n1 = 0
	} else {
		t1 *= t1
		n1 = t1 * t1 * gi1.Dot3(x1, y1, z1)
	}

	t2 := 0.6 - x2*x2 - y2*y2 - z2*z2
	if t2 < 0 {
		n2 = 0
	} else {
		t2 *= t2
		n2 = t2 * t2 * gi2.Dot3(x2, y2, z2)
	}

	t3 := 0.6 - x3*x3 - y3*y3 - z3*z3
	if t3 < 0 {
		n3 = 0
	} else {
		t3 *= t3
		n3 = t3 * t3 * gi3.Dot3(x3, y3, z3)
	}

	return 32 * (n0 + n1 + n2 + n3)
}

// Perlin noise stuff
func fade(t float64) float64 {
	return t * t * t * (t*(t*6-15) + 10)
}

func lerp(a, b, t float64) float64 {
	return (1-t)*a + t*b
}

func (n *Noise) Perlin2(x, y float64) float64 {
	// Find unit grid cell containing point
	X, Y := math.Floor(x), math.Floor(y)

	// Get relative xy coordinates of point within that cell
	x, y = x-X, y-Y

	// Wrap the integer cells at 255 (smaller integer period can be introduced here)
	xi, yi := int(X)&255, int(Y)&255

	// Calculate noise contributions from each of the four corners
	n00 := n.GradP[xi+n.Perm[yi]].Dot2(x, y)
	n01 := n.GradP[xi+n.Perm[yi+1]].Dot2(x, y-1)
	n10 := n.GradP[xi+1+n.Perm[yi]].Dot2(x-1, y)
	n11 := n.GradP[xi+1+n.Perm[yi+1]].Dot2(x-1, y-1)

	// Compute the fade curve value for x
	u := fade(x)

	// Inteporlate the four results
	return lerp(lerp(n00, n10, u), lerp(n01, n11, u), fade(y))
}

func (n *Noise) Perlin3(x, y, z float64) float64 {
	// Find unit grid cell containing point
	X, Y, Z := math.Floor(x), math.Floor(y), math.Floor(z)

	// Get relative xyz coordinates of point within that cell
	x, y, z = x-X, y-Y, z-Z

	// Wrap the integer cells at 255 (smaller integer period can be introduced here)
	xi, yi, zi := int(X)&255, int(Y)&255, int(Z)&255

	// Calculate noise contributions from each of the each corners
	n000 := n.GradP[xi+n.Perm[yi+n.Perm[zi]]].Dot3(x, y, z)
	n001 := n.GradP[xi+n.Perm[yi+n.Perm[zi+1]]].Dot3(x, y, z-1)
	n010 := n.GradP[xi+n.Perm[yi+1+n.Perm[zi]]].Dot3(x, y-1, z)
	n011 := n.GradP[xi+n.Perm[yi+1+n.Perm[zi+1]]].Dot3(x, y-1, z-1)
	n100 := n.GradP[xi+1+n.Perm[yi+n.Perm[zi]]].Dot3(x-1, y, z)
	n101 := n.GradP[xi+1+n.Perm[yi+n.Perm[zi+1]]].Dot3(x-1, y, z-1)
	n110 := n.GradP[xi+1+n.Perm[yi+1+n.Perm[zi]]].Dot3(x-1, y-1, z)
	n111 := n.GradP[xi+1+n.Perm[yi+1+n.Perm[zi+1]]].Dot3(x-1, y-1, z-1)

	// Compute the fade curve value for x,y,z
	u, v, w := fade(x), fade(y), fade(z)

	// Interpolate
	return lerp(
		lerp(
			lerp(n000, n100, u),
			lerp(n001, n101, u), w,
		),
		lerp(
			lerp(n010, n110, u),
			lerp(n011, n111, u), w,
		),
		v,
	)
}
