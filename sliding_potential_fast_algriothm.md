# Sliding Lattice Potential: Theoretical Foundation for Fast Algorithms

---

## 1. Coordinate Transformation

We consider a lattice potential sliding with velocity \( v \). The transformation to a co-moving frame is

\[ x' = x - vt, \qquad t' = t. \]

Using the chain rule, the differential operators transform as

\[ \frac{\partial}{\partial x} = \frac{\partial}{\partial x'}, \qquad \frac{\partial}{\partial t} = \frac{\partial}{\partial t'} - v \frac{\partial}{\partial x'}. \]

---

## 2. Gauge (Galilean) Transformation

Let \( m \) denote the particle mass. The wave function in the lab frame \( \psi(x,t) \) and in the moving frame \( \psi'(x',t') \) are related by the Galilean phase transformation

\[ \psi(x,t) = \exp\!\left( i\frac{m v}{\hbar}x - i\frac{m v^2}{2\hbar}t \right) \psi'(x',t'). \]

This transformation:

- adds the momentum boost \( m v \),
- shifts the energy by \( \frac{1}{2} m v^2 \),

and ensures equivalence of the Schrödinger equations in the two frames.

---

## 3. Bloch Expansion in the Sliding Frame

Let the lattice period be \( d \) and define the reciprocal lattice vector

\[ b = \frac{2\pi}{d}. \]

Consider a sliding potential of the form

\[ V \cos\!\left(\frac{2\pi}{d}x - \omega t\right), \]

where **\( \omega \) is the angular frequency** (rad/s). The phase velocity is therefore

\[ v = \frac{\omega}{b}. \]

In the sliding frame, the potential becomes stationary and Bloch's theorem applies:

\[ \psi'(x',t') = e^{-iEt'/\hbar} e^{ikx'} \sum_{n\in\mathbb{Z}} C_n e^{inbx'}. \]

Here:

- \( E \) = energy eigenvalue in the moving frame  
- \( k \) = crystal momentum (within the first Brillouin zone)  
- \( C_n \) = Fourier coefficients  

---

## 4. Transformed Wave Function in the Lab Frame

Substituting the Bloch expansion into the gauge transformation:

\[ \psi(x,t) = e^{i\frac{mv}{\hbar}x - i\frac{mv^2}{2\hbar}t} e^{-iEt/\hbar} e^{ik(x-vt)} \sum_n C_n e^{inb(x-vt)}. \]

Collecting spatial and temporal phases gives

\[ \psi(x,t) = \sum_n C_n \exp\!\left[-i t \left( \frac{E}{\hbar} + \frac{m v^2}{2\hbar} + k v + n b v \right)\right] \exp\!\left[i x \left(k + nb + \frac{m v}{\hbar}\right)\right]. \]

Using \( v = \omega/b \), we obtain

\[ \boxed{ \psi(x,t) = \sum_n C_n \exp\!\left[-i t \left( \frac{E}{\hbar} + \frac{m v^2}{2\hbar} + \frac{k\omega}{b} + n\omega \right)\right] \exp\!\left[i x \left(k + nb + \frac{m v}{\hbar}\right)\right] } \]

---

## 5. Quasi-Energy Structure

From the above expression:

- Effective crystal momentum in lab frame:
  \[ k_{\mathrm{lab}} = k + \frac{mv}{\hbar}. \]

- Effective energy:
  \[ E_{\mathrm{lab}} = E + \frac{1}{2} m v^2 + \hbar\frac{k\omega}{b} + n\hbar\omega. \]

Each Fourier index \( n \) corresponds to a **Floquet sideband** separated by \( \hbar\omega \) with spectral weight \( |C_n|^2 \).

---

## 6. Fast Algorithm for Sliding Lattice

Define the spatial evolution operator over one lattice period:

\[ F = \mathcal{X} \exp\!\left( \int_0^d G(x)\,dx \right), \]

where \( \mathcal{X} \) denotes spatial ordering.

Assume

\[ G(x) = U(x) G(0) U^\dagger(x). \]

---

### Structure of the Transformation Matrix

Define

\[ U(x) = \begin{bmatrix} u(x) & 0 \\ 0 & u(x) \end{bmatrix}, \]

with

\[ u_{pq}(x) = \delta_{pq} e^{i p b x}. \]

Indices \( p,q \in \mathbb{Z} \) label Fourier modes. (Note: indices are distinct from the particle mass \( m \).)

Define the diagonal matrix

\[ \mathrm{N} = \mathrm{diag}(p). \]

Then

\[ U^\dagger(\Delta x) = \exp(-i b \mathrm{N}\Delta x). \]

---

### Evaluation of the Ordered Exponential

Discretizing space:

\[ F = \lim_{\Delta x\to0} \prod_{j=0}^{d/\Delta x - 1} U((j+1)\Delta x) e^{G(0)\Delta x} U^\dagger((j+1)\Delta x). \]

Using

\[ U^\dagger(\Delta x) = \exp(-i b \mathrm{N}\Delta x), \]

we obtain for small \( \Delta x \):

\[ U^\dagger(\Delta x) e^{G(0)\Delta x} = \exp\!\left[ (G(0) - i b \mathrm{N})\Delta x + O(\Delta x^2) \right]. \]

Since higher-order commutators are \( O(\Delta x^2) \), they vanish in the continuum limit, yielding

\[ \boxed{ F = \exp\!\left[ (G(0) - i b \mathrm{N}) d \right]. } \]

This removes explicit spatial ordering and enables efficient computation.

---

## 7. Example: Cosine Sliding Potential

Consider

\[ V \cos\!\left(2\pi x - \omega t\right), \]

with parameters

\[ \hbar = 1, \quad m = \frac{1}{2}, \quad d = 1. \]

Let \( \epsilon \) denote the **Floquet quasi-energy parameter**.

The evolution operator becomes

\[ F(\epsilon) = \exp \left( \begin{bmatrix} -iB & I \\ M(\epsilon) & -iB \end{bmatrix} \right). \]

---

### Matrix Elements

\[ M_{pq} = -(\epsilon + p\omega)\delta_{pq} + \frac{V}{2} (\delta_{p,q+1} + \delta_{p+1,q}), \]
\[ B_{pq} = \delta_{pq} \, 2\pi p. \]

Here:

- \( I \) = identity matrix of matching dimension  
- Fourier indices \( p,q \in [-N,N] \) are truncated for numerical implementation  

---

## 8. Numerical Implementation Notes

- Truncate Fourier modes to finite range \( [-N,N] \).
- Matrix size scales as \( (2N+1) \).
- The fast algorithm avoids repeated spatial integration.
- Computational complexity is dominated by matrix exponentiation.

---

## Summary

The sliding lattice problem can be:

1. Reduced to a stationary Bloch problem in a co-moving frame.
2. Mapped to a Floquet quasi-energy spectrum in the lab frame.
3. Computed efficiently using a similarity-transformed evolution operator:

\[ F = \exp[(G(0) - i b \mathrm{N}) d]. \]

This provides both physical transparency and computational efficiency.