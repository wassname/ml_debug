Title: Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks

URL Source: https://arxiv.org/pdf/2104.08426

Published Time: Mon, 23 Jan 2023 09:31:57 GMT

Number of Pages: 50

Markdown Content:
# Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks 

N. Sukumar a, ∗, Ankit Srivastava b

> aDepartment of Civil and Environmental Engineering, University of California, Davis, CA 95616, USA
> bDepartment of Mechanical, Materials, and Aerospace Engineering, Illinois Institute of Technology, Chicago, IL 60616, USA

Abstract 

In this paper, we introduce a new approach based on distance fields to exactly impose boundary conditions in physics-informed deep neural networks. The challenges in satisfying Dirichlet boundary conditions in meshfree and particle methods are well-known. This issue is also pertinent in the development of physics informed neural networks (PINN) for the solution of partial di ff erential equations. We introduce geometry-aware trial functions in artifical neural net-works to improve the training in deep learning for partial di ff erential equations. To this end, we use concepts from constructive solid geometry (R-functions) and generalized barycentric coordinates (mean value potential fields) to construct φ(x), an approximate distance function to the boundary of a domain in Rd. To exactly impose homoge-neous Dirichlet boundary conditions, the trial function is taken as φ(x) multiplied by the PINN approximation, and its generalization via transfinite interpolation is used to a priori satisfy inhomogeneous Dirichlet (essential), Neumann (natural), and Robin boundary conditions on complex geometries. In doing so, we eliminate modeling error associated with the satisfaction of boundary conditions in a collocation method and ensure that kinematic admissibility is met pointwise in a Ritz method. With this new ansatz, the training for the neural network is simplified: sole contribution to the loss function is from the residual error at interior collocation points where the governing equation is required to be satisfied. Numerical solutions are computed using strong form collocation and Ritz minimization. To convey the main ideas and to assess the accuracy of the approach, we present numerical solutions for linear and nonlinear boundary-value problems over convex and nonconvex polygonal domains as well as over domains with curved bound-aries. Benchmark problems in one dimension for linear elasticity, advection-di ff usion, and beam bending; and in two dimensions for the steady-state heat equation, Laplace equation, biharmonic equation (Kirchho ff plate bending), and the nonlinear Eikonal equation are considered. The construction of approximate distance functions using R-functions extends to higher dimensions, and we showcase its use by solving a Poisson problem with homogeneous Dirichlet boundary conditions over the four-dimensional hypercube. The proposed approach consistently outperforms a stan-dard PINN-based collocation method, which underscores the importance of exactly (a priori) satisfying the boundary condition when constructing a loss function in PINN. This study provides a pathway for meshfree analysis to be conducted on the exact geometry without domain discretization. 

Keywords: deep learning, meshfree method, distance function, R-function, transfinite interpolation, exact geometry 

1. Introduction 

Machine learning algorithms based on supervised learning (deep neural networks) are relatively mature in fields such as computer vision, image processing, and speech recognition. Some of the earliest studies on physics-informed neural networks (PINN) to solve boundary-value problems can be traced to the contributions of Lagaris et al. [1– 3], and those of McFall [4] and McFall and Mahan [5]. These studies have provided the impetus for the recent interest and advancement of the approach. Over the past 3–4 years there have been many new developments in a meshfree approach that is based on PINN to solve low-and high-order partial di ff erential equations (PDEs). Some of the main contributions in this thread are based on: collocation [6–8], variational principle (deep Ritz) [9, 10], and Petrov-Galerkin domain-decomposition [11, 12]. Lu et al. [13] present an overview of PINN for the solution    

> ∗Corresponding authors
> Email address: nsukumar@ucdavis.edu (N. Sukumar)
> To Appear in Computer Methods in Applied Mechanics and Engineering November 9, 2021
> arXiv:2104.08426v2 [math.NA] 7 Nov 2021

of PDEs. These developments have been made possible due to the advances in automatic di ff erentiation tools that effi ciently compute the derivatives of nonlinear composite functions, and in stochastic gradient descent algorithms that can deliver accurate solutions for nonlinear, nonconvex optimization problems. Furthermore, the availability of public-domain data analysis packages such as Tensorflow [14] and PyTorch [15] has also decreased the barriers to entry for newcomers to this field. Within the purview of solving PDEs with neural networks, it is instructive to highlight that unlike well-known computational methods (finite elements and its generalizations, finite volume, boundary elements, meshfree, and others), where function approximation is based on linear combinations of basis functions, which have to be chosen and defined a priori, function approximation in deep neural networks is through nonlinear function composition of an activation function σ : Rd → R, which yields the best approximation function via the solution procedure. A popular choice for σ is: σ(x) : = ReLU (x) = max(0 , x), which is known as the Rectified Linear Unit (ReLU) activation function. The collocation method using PINN is based on minimizing the least squares residual error (nonlinear and nonconvex mean squared error loss function) in order to satisfy the PDE and the boundary conditions at collocation points—whose solution yields the parameters that define the approximation function in the artificial neural network. In PINN-based deep collocation [6] and deep Ritz [9] methods, the inexact imposition of boundary conditions adversely a ff ects the training of the neural network as well as the accuracy of the method [16–18]. On complex ge-ometries, this shortcoming becomes more acute. This issue has plagued meshfree Galerkin methods [19, 20] since their inception, and a foolproof solution is still unavailable for arbitrary two- and three-dimensional geometries. Meshfree basis functions are also nonpolynomial, and hence an additional hurdle is a suitable cell-based numerical integration scheme that is consistent (patch test is passed) and stable (absence of spurious modes) for nonlinear simulations. Since solving for the approximating function is part of the minimization procedure in PINN, a simple (e.g., Monte Carlo) integration scheme for the entire domain can be used. Hence, if a reliable approach to impose boundary conditions on complex geometries in PINN is realized, it can lead to an accurate and robust meshfree method. In this paper, we tackle this problem by proposing a new approach in PINN to exactly impose boundary conditions, with an eye towards enabling meshfree simulations over complex geometries in Rd (d = 2, 3). Our main contributions are as follows: 1. We introduce a new geometry-aware method in physics-informed neural networks that uses R-functions and transfinite interpolation to exactly impose boundary conditions over complex (a ffi ne, curved and multiply-connected) geometries. This geometry-aware approach, which is based on the construction of approximation distance functions (ADFs) to boundary sets, was first proposed by Kantorovich to satisfy Dirichlet boundary conditions [21], and has been extended using the theory of R-functions to exactly impose Dirichlet, Neumann and Robin boundary conditions in a meshfree method by initially Rvachev [22], and then subsequently by Rvachev and coworkers [23–25] and Shapiro and coworkers [26–31]. A related meshfree approach in the spirit of Kantorovich’s method is that of H¨ ollig et al. [32], who used web-splines that are multiplied by a weight func-tion (approximates the distance function) to solve a Dirichlet problem. We also draw from a previous study [33], where R-functions are used to define smooth approximate distance fields over polygonal domains. 2. Approximate distance fields that stem from mean value potentials [34–36] are also used in PINN to solve PDEs. 3. On exactly satisfying boundary conditions in physics-informed neural networks, the training of the network is simplified, and this facilitates convergence and improved accuracy of the PINN approximation. 4. New application of R-functions in R4: solution of the Poisson equation over the four-dimensional hypercube. 5. Solving PDEs on curved domains without discretizing the domain (mesh generation) is realized, which provides a pathway to conducting meshfree simulations on the exact geometry (isogeometric analysis) [37]. First, we present a few connections of prior work on finite elements and partition-of-unity meshfree methods to better understand and place the PINN approximation and its use to solve PDEs. Contributions in r-adaptivity with finite elements have a long history in computational mechanics. For instance, in finite-deformation simulations using finite elements, the optimal nodal locations and the solution coe ffi cients have both been simultaneously treated as unknowns in the minimization of the potential energy functional [38]. Since the PINN approximation that is composed by the ReLU activation function can exactly represent piecewise a ffi ne functions (Delaunay basis functions) [39], one can view the ReLU network solution as a variational r-adaptive finite element solution procedure. Instead of refining elements in h-adaptive finite elements, adaptive solutions can be realized via a basis refinement strategy that has advantages (for example, “hanging nodes” are a nonissue), which was put forth by Grinspun [40], and a similar approximation refinement perspective can be associated with a multilayer neural network [41]. The connections between ReLU networks and hp -finite elements are studied in Opschoor et al. [42]. Initiated by Kansa [43, 44], meshfree collocation schemes with positive-definite radial basis functions (RBFs) such as the Gaussian and Hardy’s 2inverse multiquadrics have been used to solve PDEs [45, 46]. Schaback and Wendland [47] discuss the ties of kernel methods (for example, radial basis functions) to machine learning and meshfree methods. The choice on how to set the “shape parameter” in the Gaussian RBF (controls the support-width of the Gaussian) is an unsettled issue since it is problem dependent and must be carefully selected for boundary and interior points if exponential convergence is to be maintained without exacerbating the condition number. A related approach is the local maximum-entropy (max-ent) meshfree method [48] that yield compactly-supported basis functions of exponential form that constitute a partition-of-unity [49], possess linear completeness, and provide a smooth transition from Delaunay basis functions [50] to global maximum-entropy basis functions [51]. Consider a nodal set with nodes (centers) that are located at {xi}ni=1. When viewed through the lens of Gaussian weight functions [52–54], a single parameter {βi}ni=1 controls the support-width of each Gaussian weight function. Rosolen et al. [55] proposed a variational adaptivity formulation to find optimal values of {βi}ni=1 that minimize the potential energy functional for Poisson equation and nonlinear elasticity. Since RBFs can be represented in a neural network with a single hidden layer [56, 57], the neural network solution optimizes the location of the centers {xi}ni=1 as well as the support-widths {βi}ni=1 [56]. RBF-based partition-of-unity networks [58] for hp -approximation have been introduced, and numerical experiments have been conducted using sparse Gaussian networks to solve PDEs [59]. Lastly, we mention the recent work of Greco and Arroyo [60], who presented a collocation scheme for PDEs that is based on high-order max-ent approximants, which delivered accurate simulation results on domains with a ffi ne and curved boundaries. In Section 9.1.7, we present a one-dimensional example using Gaussian neural networks. Since PINN a ff ords significant flexibility vis-` a-vis existing meshfree (basis-set) methods, solving PDEs over complex geometries using collocation and Ritz methods with artificial neural networks holds significant promise. In a strong collocation PINN method, the loss function consists of the residual errors from the interior of the domain, which is known as interior or PDE loss , and from the boundaries of the domain, which is referred to as 

boundary (conditions) loss [6]. There are three distinct contributions in the mean squared error loss function: (1) residual error at interior collocation points where the PDE must be satisfied; (2) residual error at boundary collocation points where the essential (Dirichlet) boundary condition must be satisfied; and (3) residual error at boundary collo-cation points where the Robin or natural (Neumann) boundary condition must be satisfied. Early approaches [1, 3–5] had already recognized the importance of exact imposition of boundary conditions in artificial neural networks. La-garis et al. [1] considered two terms in the trial function, with the first term being an analytical function that exactly imposed the boundary conditions and the second term was chosen as the product of the PINN approximation and a function that vanished on the boundary; for irregular boundaries, Lagaris et al. [3] used a RBF network in the first term to satisfy the boundary conditions at a collection of discrete points on the boundary. McFall et al. [4, 5] and more recently Sheng and Yang [61] introduced a length factor (measure of the distance to the boundary) associated with the boundary to impose boundary conditions, and Berg and Nystr¨ om [7] approximated the distance function using a low-capacity neural network to impose boundary conditions over complex geometries. In many recent stud-ies [16–18, 62], the implications of imposing essential boundary conditions via the loss function in PINN have been studied, and numerical experiments have a ffi rmed that the presence of the boundary residual terms compromises the convergence of the stochastic gradient descent algorithm and the accuracy of the method. To address this problem, remedies have been introduced in the PINN literature, such as using two neural networks, one for the PDE and the other to satisfy the essential boundary condition [3, 7, 61–63], introduction of a penalty parameter via an augmented variational formulation to weakly impose the essential boundary conditions [9], and Nitsche’s method to impose the essential boundary condition [64]. Some of these approaches mirror those previously pursued in meshfree and particle methods to satisfy essential boundary conditions [19, 20, 65]. In meshfree Galerkin methods, the choice of the space for Lagrange multipliers is delicate; penalty approach leads to a saddle-point problem and the inf-sup condition must be met; and though Nitsche’s method is variationally consistent, the stabilization parameter in it must be judiciously chosen. For low-dimensional problems over complex geometries, an accurate and robust meshfree approach remains elusive. Since from the universal approximation theorem [66, 67] we know that a neural network with one hidden layer can represent any L2 function to arbitrary accuracy, it stands to choose an ansatz that satisfies the boundary conditions a priori, so that the loss function is expressed solely in terms of the residual error at only the interior collocation points where the PDE is required to be satisfied. If the essential boundary conditions are exactly met, then this precludes “variational crimes” in a Ritz method [68]. Lastly, and most importantly, in deep collocation [6], multiple terms (inte-rior loss and boundary losses) that have to be individually minimized are incorporated within a single objective (loss) function. When this loss function is minimized, then the solution that is realized depends on the weight (equal weights is the unbiased choice) that is assigned to each objective function, which reflects the importance of each residual error contribution. Rohrhofer et al. [69] discuss network training in relation to the Pareto front that appears in multiobjec-tive constrained optimization. In the NVIDIA SimNet ™ toolkit [70], signed distance function weighting is used to 3dynamically assign the spatially varying weight functions for the PDE and boundary loss terms. Since these weights are problem dependent, they should not be fixed a priori, since then the magnitude of the training loss by itself is a misleading error measure. To establish the accuracy of the method, the error in u as well as in u′ must be assessed. We present numerical results in Sections 9–11 that support this thesis, and which points to the merits of the new approach. In this paper, we solve this problem of competing loss terms in PINN formulations by constructing a trial function for the neural network that a priori satisfies all boundary conditions in deep collocation, and meets kinematic admis-sibility when used in a deep Ritz method. This eliminates the boundary terms in the loss function. Our approach is based on constructing distance functions (exact or approximate) to the boundary of the domain, and it can treat essen-tial (Dirichlet) as well as mixed (Dirichlet and Robin) boundary conditions over complex domains. We use the exact distance function whenever it is available and applicable. However, in general, we construct approximate distance functions using two di ff erent techniques: the theory of R-functions [23, 30]) and the theory of mean value potential fields [34–36]. These methods provide approximate distance functions that possess the desirable property of being zero on the boundary of the domain with unit (inward) normal directional derivative. In addition, they are smooth in the interior of the domain, a property that the exact distance function does not always possess. Functions whose sign solely depend on the sign of its arguments encode Boolean logic, and are known as R-functions. R-functions provide an implicit function representation for line segments, curves, and solid regions, and are composed by Boolean operations (negation, conjunction, disjunction, equivalence). Mean value potential fields are specific forms of a sin-gular double-layer potential that yield Lp-distance fields [36]. For a domain in R2, this singular potential is defined as the integral of the reciprocal of the p-th power of the distance from its boundary. For a polygon with p = 1, closed-form expressions for the ADF are available [34], but for closed curves in R2, numerical integration is required to compute the ADF. Once the approximate distance functions are formed, methods to impose essential and Robin boundary conditions are available that rely on transfinite interpolation [23, 25]. R-functions with approximants such as B-splines [27, 31] and RBFs [71] have been used in a meshfree Galerkin method to solve boundary-value problems. The remainder of this paper is organized as follows. In Section 2, we discuss the properties of distance functions, and in Section 3, the essentials on R-functions and the construction of approximate distance functions are described. In particular, joining R-functions using R-equivalence composition is presented, which is used in this paper. The inverse of the normalizing weight function that appears in the expression for mean value coordinates (polygon) and transfinite mean value interpolant (closed curves), is an approximate distance field. These are particular instances ( p = 1) of 

Lp-distance fields, and are discussed in Section 4. On using normalizing functions and solution structures in the R-function method [22, 23], we describe in Section 5 the use of ADFs to construct an ansatz in PINN that exactly satisfies boundary conditions for second- and fourth-order problems. The construction of the trial function in a deep neural network is presented in Section 6, along with a summary of the feedforward neural network and the backpropagation (computation of the gradient of the loss function and use of stochastic gradient descent) algorithm. The loss function for collocation and deep Ritz formulations are presented in Section 7. The numerical implementation is discussed in Section 8, where we also provide code snippets of some of the main functions. One- and two-dimensional numerical simulations are presented in Sections 9 and 10, where we apply this new approach to a broad suite of boundary-value problems (Poisson, harmonic coordinates, plate bending, Eikonal equation) on convex and nonconvex domains with a ffi ne and curved boundaries. In addition, the Poisson equation over the four-dimensional hypercube is solved in Section 11. These numerical results clearly demonstrate the benefits of exactly imposing the boundary conditions in PINN—it simplifies the training of the network and enhances the accuracy and robustness of the method. Finally, we conclude with Section 12, where we summarize the main developments in this paper and discuss some of the topics of future research. 

2. Distance Functions and their Properties 

The signed distance function is an implicit representation for curves and surfaces, and also provides fast evaluation of predicates for geometric objects. Let S ⊂ Rd denote a domain (open, bounded set) with boundary ∂S . The exact distance function d(x) gives the shortest distance between any point x ∈ Rd to ∂S . It is clear that d(x) is identically zero on ∂S . Computing the exact distance function requires solving the Eikonal equation (see Section 10.5), which is computationally expensive. Therefore, it is desirable to construct an approximate distance function or ADF (formally represented by φ(x)) that has a closed-form (non-iterative algorithm) expression. Furthermore, since the exact distance function may only be continuous and not continuously di ff erentiable, it may not be suitable for use in a trial function to solve PDEs. Since our objective is to use ADFs in a collocation or Ritz method to solve boundary-value problems, their di ff erential properties are important. If essential boundary conditions are imposed on the entire boundary ∂S ,4then the ADF must be zero on ∂S , positive in S , and its gradient must not vanish for any x ∈ ∂S . In addition, since the exact distance function has derivative discontinuities on the medial axis of the domain, smooth approximations of the distance function must be used within the trial function for a collocation method with PINN. For a second-order problem, a C0 distance function that has gradient discontinuities in the interior of the domain cannot be used in the collocation approach since the Laplacian of the distance function will be unbounded at a collocation point. These considerations are crucial when used to solve PDEs. For instance, positivity in S precludes the presence of singularities within the domain, which in general is di ffi cult to construct as noted in McFall [4]. For clarity of exposition in this paper, we use ν := ν(x) to denote the unit inward normal vector (appears in the theory related to R-functions) on the boundary ∂S , and n := n(x) as the unit outward normal vector (used when defining Neumann or Robin boundary condition) on ∂S . It is noted that n = −ν. If ∂S in R2 is composed of piecewise line segments and curves, then we use φi := φi(x) to denote the ADF to each curve or line segment. For a point x ∈ Rd

on ∂S , it is essential that any approximation to the distance function satisfy φ = 0. Furthermore, to mimic the exact distance function, the normal derivative with respect to ν on the boundary should be unity, ∂d/∂ν = ∇d · ν = 1, and it is desirable that all higher order normal derivatives vanish. An m-th order approximate distance function requires that the second- to m-th order normal derivatives vanish on all regular points (unit normal is well-defined) on ∂S [23]: 

∂φ ∂ν = 1, ∂kφ∂ν k = 0 (k = 2, 3, . . . , m), (1) and such a function is said to be normalized to the m-th order. For finite m, the normalized function matches the exact distance function only in the vicinity of the boundary; for points that are away from the boundary, it deviates from the exact distance. Apart from applications in solid modeling, mesh generation, real-time rendering and computer vision, where distance functions are used, normalized first-order distance functions are also a suitable choice for the initialization and assignment of the extension velocity at points away from the interface in the level set method [72]. As noted in Biswas and Shapiro [29], use of normalized distance functions mitigate the bulging phenomenon in the vicinity of where the segments or curves are joined [73], since undulations (presence of local extrema) are undesirable in the representation of the surface. 

3. R-functions and Approximate Distance Functions 

The theory of R-functions can be used to construct a composite approximate distance function, φ(x), to any arbi-trarily complex boundary ∂S , when approximate distance functions, φi(x), to the partitions of ∂S are known. Consider a real-valued function F(ω1, ω 2, . . . , ω q), where ωi(x) : Rd → R (i = 1, . . . , q) are also real-valued functions. If the sign of F(·) is solely determined by the signs of its arguments ωi(x), then F(·) is known as an R-function [26, 30]. R-functions were proposed by T. L. Rvachev in 1963 [26]. For example, F1(x, y) = 1 + x2 + y2 and F2(x, y) = xy 

are R-functions in R2, whereas F3(x, y) = √x2 + y2 − 1 and F4(x, y) = sin xy are not. The important properties of R-functions are provided in Rvachev and Sheiko [23] and Shapiro [26]. On combining set-theoretic Boolean operations with such functions, the inverse problems of semi-analytic geometry (solid modeling) can be solved. Consider a continuous function ωi : Rd → R. Let Ω ⊂ Rd be an open, bounded domain, ¯ Ω = Ω ∪ ∂Ω be the closure of Ω, and define Ωc to be the complement of Ω (Ω ∪ Ωc = Rd). If ωi is strictly positive in Ω, identically equal to zero on ∂Ω, and strictly negative in Ωc\∂Ω, then it is evident that F(ωi) = ωi is an R-function. Over the region ¯ Ω, we associate ωi with the Boolean 1 (logical true) and over the region Ωc we associate it with the Boolean 0 (logical false). Note that ωi = 0 (0 is assumed to be signed) is included in both sets so that it can be assigned to either the set of negative real values or the set of positive real values [30]. Hence, similar to Boolean functions, ωi is closed under composition. Furthermore, just as Boolean functions are written using the symbols ¬ ∨ , and ∧, which correspond to complement, union and intersection in set theory, every R-function can be written as the composition of the corresponding elementary R-functions: R-negation ( −ω), R-disjunction ( ω1 ∨ ω2), and R-conjunction ( ω1 ∧ ω2). On defining R-functions for regions in Rd, a solid can then be composed using the set-theoretic operations of ¬, ∨,and ∧. For the universal set U = R2, Venn diagrams for some of the operations in set theory are shown in Fig. 1, and the corresponding operations using R-functions are indicated. The simplest examples of R-functions are the R-disjunction (union) and the R-conjunction (intersection) functions. These are 

ω1 ∨ ω2 = ω1 + ω2

2 +

√(ω1 − ω2)2

2 = max( ω1, ω 2), ω1 ∧ ω2 = ω1 + ω2

2 −

√(ω1 − ω2)2

2 = min( ω1, ω 2).

5A B(a) A B (b) A B (c) A B (d)                                     

> Figure 1: Venn diagram for union, intersection, complement and equivalence in R2.ωAand ωBare R-functions that are positive in ΩAand ΩB
> (open sets), respectively. (a) A∪B≡ωA∨ωB; (b) A∩B≡ωA∧ωB; (c) ¯ A≡ − ωA; and (d) ( A∩B)∪( ¯ A∩¯B)≡ωA∼ωB. Examples of ∨and ∧
> operations using R-functions are given in (2) and (3).

The generalization of the above R-functions is [30]: 

Rα(ω1, ω 2) : = 11 + α

(

ω1 + ω2 ±

√

ω21 + ω22 − 2αω 1ω2

)

, (2) with ( +) and ( −) signs defining R-disjunction and R-conjunction, respectively. If ω1 and ω2 denote the sides of a triangle, then the triangle inequality is expressed in (2) with −1 < α < 1 being the cosine of the angle between the two sides. For α = 1, the max and min R-functions are recovered. If ω1 and ω2 are positive, then so are ω1 ∨ ω2 and 

ω1 ∧ ω2. The R-functions defined in (2) are not analytic at points where ω1 = ω2 = 0. Smoothness can be obtained by defining the function ( α = 0 is selected) [30] 

Rs(ω1, ω 2) : =

[

ω1 + ω2 ±

√

ω21 + ω22

] ( ω21 + ω22

) s 

> 2

, (3) which renders these functions to be Cs-continuous at all points other than where ω1 = ω2 = 0. 

3.1. Normalized functions for line segments and curves 

Shapiro and Tsukanov [74] describe the representation of line segments and curves using R-functions and discuss their di ff erential properties. Let us consider one line segment that joins x1 := (x1, y1) and x2 := (x2, y2). The center of this segment is denoted by xc := (x1 + x2)/2, and the length of the segment is: L = || x2 − x1|| . Now, we define [25] 

f := f (x) = (x − x1)( y2 − y1) − (y − y1)( x2 − x1)

L , (4) which is the signed distance function from point x to the line that passes through x1 and x2.Since the representation of the segment can be viewed as the intersection of an infinite line with a disk of radius 

L/2, we consider the following trimming function that is normalized to first order [25]: 

t := t(x) = 1

L

[( L

2

)2

− || x − xc|| 2

]

, (5) where t ≥ 0 defines a disk with center at xc. Now, with f (x) and t(x) on-hand, we define a normalized function (up to first order) φ(x) that is C2 at all points away from the line segment [25, 29]: 

φ := φ(x) =

√

f 2 +

( ϕ − t

2

)2

, ϕ =

√

t2 + f 4, (6) which is an approximation of the distance function to the segment with end points x1 and x2. The function φ in (6) is a modification of the form ϕ = |t| [74], which has a derivative discontinuity at t = 0. Figure 2 provides a graphical illustration of f , t and φ for a line segment. For a quarter-circular arc, the functions 

f , t and φ are shown in Fig. 3. In Fig. 4, the approximate distance function (normalized to order 1) to a circle and an ellipse are presented. The ADF to a circle of radius R and center located at xc := (xc, yc) is given by 

φ(x) = R2 − (x − xc) · (x − xc)2R , (7) 6-1 -0.5 0 0.5 1-0.5 00.5         

> -0.5 00.5 -1 -0.5 00.5 1-0.5 00.5
> -1 -0.5 0-1 -0.5 00.5 1-0.5 00.5
> 00.5 1

Figure 2: Construction of the approximate distance function to a line segment. The leftmost plot depicts the signed distance function (4) to a line in 

R2; the middle plot shows the trimming function (5); and the rightmost plot displays the approximate distance function (6) to a line segment. 0.5 1 1.5 2-0.5 00.5 1      

> -1.6 -1.4 -1.2 -1 -0.8 -0.6 -0.4 -0.2 00.2 0.5 11.5 2-0.5 00.5 1
> -1 -0.5 00.5 10.5 11.5 2-0.5 00.5 1
> 0.2 0.4 0.6 0.8 11.2 1.4 1.6 1.8

Figure 3: Construction of the approximate distance function to a quarter-circular arc. The leftmost plot depicts f , the equation of the circular arc (normalized to first order); the trimming function is shown in the middle plot; and the rightmost plot displays the approximate distance function given by (6). 

where φ(x) is a smooth (bivariate polynomial of degree 2) function. For an elliptical disk whose closure (interior and boundary) is given by the R-function ω(x) ≥ 0, we construct an ADF that is normalized to order 1 using [23] 

φ(x) = ω(x)

√ω2(x) + ||∇ ω(x)|| 2 . (8) -1 -0.5 0 0.5 1-1 -0.5 00.5 1

00.2 0.4 0.6 0.8 

> (a)

-1 -0.5 0 0.5 1-1 -0.5 00.5 1

00.2 0.4 0.6 0.8 (b) 

Figure 4: Approximate distance function (normalized to order 1) to a (a) circle and an (b) ellipse. 

In general for curves that are given in parametric form, such as B´ ezier and non-uniform rational B-spline (NURBS) curves, constructing ADFs require implicitization of the curve. These extensions are discussed and presented in Upreti et al. [75]. For some of the considerations and challenges in the representation (implicit and parametric) of curves in enriched computational methods, see Chin and Sukumar [76]. 73.2. R-equivalence operation 

Given the normalized distance functions φ1 and φ2 for two curves c1 and c2, a distance field φ(φ1, φ 2) for the union 

c1 ∪ c2 must be zero when either φ1 = 0 or φ2 = 0 and positive otherwise. When c1 and c2 are line segments, the naive formula φ(φ1, φ 2) = φ1φ2 is no longer normalized at the regular points of the segments. An R-equivalence solution that preserves normalization up to order m of the distance function at all regular points (nonvertices for polygonal curves) is given by [29]: 

φ(φ1, φ 2) : = φ1 ∼ φ2 = φ1φ2

> m

√φm 

> 1

+ φm

> 2

= 1

> m

√ 1

> φm
> 1

+ 1

> φm
> 2

, (9) where lim m→∞ φ(φ1, φ 2) = min( φ1, φ 2). When ∂S (closed curve) is composed of n pieces, then a φ that is normalized up to order m is given by (see the proof in [33]): 

φ(φ1, . . . , φ n) : = φ1 ∼ φ2 ∼ · · · ∼ φn = 1

> m

√ 1

> φm
> 1

+ 1

> φm
> 2

+ . . . + 1

> φmn

. (10) The φ that is formed in (10) can be viewed as the reciprocal of the Lm-norm of inverse distance measures, which bears similarity to Lm-distance fields [36]. An alternative joining procedure is to consider the R-conjunction given by 

φs(φ1, φ 2) : = φ1 ∧ φ2 = φ1 + φ2 − s

√

φs 

> 1

+ φs

> 2

, (11) which is a function that is normalized to the ( s − 1)-th order [29]. However, the joining operation is not associative, which makes this choice less desirable. The R-equivalence joining relation in (10) is associative. The approximate distance function to two line segments is shown in Fig. 5, where the R-conjunction composition in (11) with s = 2, 3 and the R-equivalence relation in (9) with m = 1, 2 are compared. The bulging phenomenon [73] is noticeable in the vicinity of the joining point. In Fig. 6, we present the approximate distance function to a curved triangular region using R-equivalence for di ff erent orders of the normalizing parameter m. As m increases, the ADF approaches the exact distance, which is observed when inspecting the contours in the interior of the curved triangle. The ADFs for a triangle, square, hexagon, and an L-shaped polygon are shown in Fig. 7. The ADF for a square in Fig. 7b bears similarities to a superellipse, |x|m + |y|m = 1, which has rounded corners as m increases. As the last example, we present the ADF for a complex polygonal domain. We consider the polygonalized map of Bhutan, 1

which has 291 boundary vertices. The contour plot for φ(x) is depicted in Fig. 8, and we observe that the contours are smooth in the interior and well-separated ( φ is monotonic in Ω). Finally, we mention that a modified form of the R-equivalence relation (10) is also discussed in Biswas and Shapiro [29]. This modified form is constructed with an eye on better capturing the first-order normalization condition in the sector region between two line segments, where the normal is undefined and the closest point to either segment is the common vertex. With increase in m, the R-equivalence joining operation provides a better approximation to the exact distance away from the segments and also improved normalization properties in the vicinity of the segments; however, this comes at the expense of the function being higher order and hence its Laplacian will have greater undulations. Use of m = 2 has been adopted in prior computational studies [29, 33, 75], but herein, we adopt m = 1 in most of the numerical simulations that are presented in Sections 9–11. 

4. Generalized Mean Value Potentials and Lp-Distance Fields 

In addition to the theory of R-functions, another approach to construct approximate distance fields is via the theory of mean value potential fields. This has connections to generalized barycentric coordinates and in particular to mean value interpolation over polygons and curved domains [77]. Generalized barycentric coordinates [78–80] are an extension of barycentric coordinates over simplices to polygons and polyhedra. These coordinates (shape functions) have linear precision and are nonnegative over convex polygons. Transfinite barycentric interpolation over domains bounded by curves is the continuous counterpart of generalized barycentric coordinates over polygons [77]. Given a function u : R2 → R that assumes the function g(x) on the boundary (curved) of a domain, a transfinite interpolant provides an approximation of u(x) that matches g(x) over the curved boundary of the domain. For a domain that    

> 1Vectorized eps image obtained from https://freevectormaps.com

8-1 -0.5 0 0.5 1-1 -0.5 00.5 1    

> 0.2 0.4 0.6 0.8 11.2 1.4 -1 -0.5 00.5 1-1 -0.5 00.5 1
> 0.2 0.4 0.6 0.8 11.2 1.4 1.6 1.8

(a) -1 -0.5 0 0.5 1-1 -0.5 00.5 1     

> 0.2 0.4 0.6 0.8 11.2 -1 -0.5 00.5 1-1 -0.5 00.5 1
> 0.2 0.4 0.6 0.8 11.2 1.4 1.6

(b) 

Figure 5: Approximation of the distance function to two line segments. (a) R-conjunction composition with s = 2, 3 in (11), and (b) R-equivalence composition in (9) for the normalization parameter m = 1, 2. The ADFs are normalized to order s − 1 and m, respectively. 0.5 1 1.5 2-0.5 00.5 1

> 0.2 0.4 0.6 0.8 11.2 1.4

(a) m = 20.5 1 1.5 2-0.5 00.5 1 

> 0.2 0.4 0.6 0.8 11.2 1.4 1.6

(b) m = 30.5 1 1.5 2-0.5 00.5 1 

> 0.2 0.4 0.6 0.8 11.2 1.4 1.6 1.8

(c) m = 60.5 1 1.5 2-0.5 00.5 1 

> 0.5 11.5 2

(d) m = 10 

Figure 6: Plots of the approximate distance function to a curved triangle using R-equivalence for di ff erent choices of the normalizing parameter (m = 2, 3, 6, 10). -1 -0.5 0 0.5 1-1 -0.5 00.5 1

> 0.1 0.2 0.3 0.4 0.5 0.6 0.7

(a) -1 -0.5 0 0.5 1-1 -0.5 00.5 1 

> 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4

(b) -1 -0.5 0 0.5 1-1 -0.5 00.5 1 

> 0.1 0.2 0.3 0.4 0.5 0.6

(c) -1 -0.5 0 0.5 1-1 -0.5 00.5 1 

> 00.1 0.2 0.3 0.4 0.5

(d) 

Figure 7: Plots of the approximate distance function using R-equivalence ( m = 1) for polygons. (a) triangle, (b) square, (c) regular hexagon, and (d) L-shaped (nonconvex) polygon. 100 200 300 400 500 600 100 200 300 400 

20 40 60 80 100 

Figure 8: Plot of the approximate distance function using R-equivalence ( m = 1) for polygonalized map of Bhutan. The polygon has 291 vertices. 

9is bounded by a ffi ne or curved boundaries, the reciprocal of the mean value normalization function is a smoothed approximation to the exact distance function [35, 81, 82], and is a specific instance ( p = 1) of the reciprocal of a singular double-layer Lp-potential field [36]. We refer to the method that generates these smoothed distance functions by the acronym MVP, since they stem from (generalized) mean value potential fields [77]. The construction of φ(x)over polygons and curved domains is presented in Sections 4.1 and 4.2, respectively. 

4.1. Approximate distance fields on arbitrary planar polygons 

A popular generalized barycentric coordinate is Floater’s mean value coordinates [34], which is derived using the circumferential mean value theorem for harmonic functions. This conception stemmed from the objective to approximate a harmonic map by a convex combination map (positive weights) over a triangulation, so that injectivity is preserved. Mean value coordinates have many remarkable properties: for instance, they are valid on arbitrary planar polygons, including nested polygons; C∞ smooth in Ω with derivative discontinuities only at the vertices of the polygon; they reduce to piecewise a ffi ne functions on the edges of a polygon; are nonnegative in the kernel of the polygon; reciprocal of the normalizing weight function is a smoothed ADF; and they also have a smooth extension outside the polygon [81]. Consider the nonconvex polygon ( n-gon) shown in Fig. 9, whose n vertices are defined in counterclockwise orien-tation. The coordinates of the vertices are {xi}ni=1, and x is an arbitrary point in the interior of the polygon. The mean value coordinates, {ϕi(x)}ni=1, are defined as [34]: 

ϕi(x) = wi(x)

W(x) , wi(x) : = tan ( αi−1/2) + tan ( αi/2) 

|| xi − x|| , W(x) =

> n

∑

> j=1

w j(x), (12) where the angles αi−1 and αi are shown in Fig. 9. Let ri := xi − x with ri = || xi − x|| represent the Euclidean distance between x and xi. On noting the half-angle formula for tan( ·), we can define 

ti := tan 

( αi

2

)

= sin αi

1 + cos αi

= riri+1 sin αi

riri+1 + ri · ri+1

= det ( ri, ri+1)

riri+1 + ri · ri+1

, W(x) =

> n

∑

> i=1

( 1

ri

+ 1

ri+1

)

ti (rn+1 := r1), (13) which is now valid for all points x that are in the interior of a convex or nonconvex polygon. The denominator vanishes when αi = π, i.e., when x lies on the boundary of the polygon, but there ϕi(x) are known. The singularity of the weight function on the boundary is a property shared by nonnegative generalized barycentric coordinates. i

# rri−1 

i

# αα

ii−1 r

i+1 

# xxi−1 

# xxi+1 

> (a)

# θi − θ y( ) 

# xrx

i+1 ii

# x rθ

# ri+1 

# θα 

> (b)
> Figure 9: Notation used in the definition of (a) mean value coordinates [34] and (b) generalized mean value potentials [77]. In (b), the parameters that are used to form W(x) in (13) are shown.

For a polygon with one interior (nested) m-gon, the vertices of the inner polygon are defined in clockwise orien-tation [81]. The contributions of {wi(x)}n+mi=1 are used to form W(x) in (12). Hormann and Floater [81] showed that 10 φ(x) = 1/W(x) is an ADF to the boundary of the polygon, where its normal derivative is 1 /2. On taking 

φ(x) = 2

W(x) , (14) where the scaling factor is the volume of the unit sphere in Rd−1 [82] (equal to 2 when d = 2), the normal derivative becomes ∂φ/∂ν = 1 on ∂Ω. As we discuss in Section 4.2, the W(x) that appears in (13) and (14) is a particular instance (p = 1) of the mean value potential field Wp(x). We now have a smooth ADF for a polygon that is normalized to order 1. In Fig. 10, the surface and contour plots of φ(x) are shown for square and L-shaped domains, as well as nested squares and octagons. The ADF using R-equivalence for the polygonalized map of Bhutan is presented in Fig. 8. In Fig. 11, we show the approximate distance field (surface and contour plots) for the same polygonalized map. 00.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 

(a) 00.02 0.04 0.06 0.08 0.1 0.12 (b) 00.02 0.04 0.06 0.08 0.1 0.12 0.14 

(c) (d) 

Figure 10: Approximate distance fields for polygonal domains. (a) square, (b) L-shaped, (c) nested squares, and (d) nested octagons. 

(a) (b) 

Figure 11: Approximate distance field for the polygonalized map of Bhutan. Surface plot is shown in (a) and the contour plot in (b). The polygon has 291 vertices. 

11 4.2. Approximate distance fields over curved domains 

Consider an open, bounded nonconvex domain Ω with boundary Γ = ∂Ω. Given a function g : Γ → R that is prescribed on a curved boundary, the transfinite mean value interpolant u : Ω → R is defined as [35, 83] 

u(x) =

∫ 

> Sv

g(y(x, v))K(x, y) dS v

W(x) , W(x) =

∫ 

> Sv

K(x, v) dS v, K(x, v) = 1

‖v − x‖ , (15) where x ∈ Ω\S v and v ∈ S v, S v is the unit circle that is centered at x, the ray from x that passes through v intersects the boundary Γ at y, and K(x, v) is a singular kernel function [35, 83]. Similar to the behavior of the inverse weight in mean value coordinates on polygons, the function φ(x) = 1/W(x)behaves like an approximate distance function to the boundary and its normal derivative on the boundary Γ is 1 /2 [35]. Belyaev et al. [36] introduced Lp-distance fields ( p ≥ 1), which approximates the exact distance function. These distance fields stem from a particular form of a singular double-layer potential, and hence the reference to them as generalized mean value potential fields. Consider the nonconvex domain shown in Fig. 12. The parametrization of the curved boundary Γ : [0 , 1] → R is c(t), and its tangent is c′(t). For a nonconvex domain, a ray from x intersects the boundary at multiple points c(ti) : = yi(x, v). On projecting the boundary curve onto the unit circle, an expression for 

φ(x) that is valid for convex as well as nonconvex domains is obtained in terms of the curve parameter t ∈ [0 , 1] [83]: 

φ(x) =

( 1

Wp(x)

)1/p

, Wp(x) =

∫ 10

(c(t) − x) · c′⊥ (t)

‖c(t) − x‖2+p dt , (16) where c′⊥ (t) : = rot (c′(t)) is obtained by rotating c′(t) through 90 ◦ in the clockwise direction. For x ∈ Ω, the integral in (16) is numerically integrated; if x ∈ ∂Ω (integral is singular), we set φ(x) = 0. In (16), Wp(x) is the generalized mean value potential field, which is used to form the approximate distance function φ(x). Equation (16) is also applicable for polygons: on choosing p = 1 in (16), we recover the W(x) that appears in (13). The approximate distance fields ( p = 1) for an elliptical disk, annulus, hypocycloid, and a propeller-shaped domain are shown in Fig. 13. The distance function is smooth in the interior of the domain, φ ∈ C∞(Ω), and it is Ck on the boundary for a Ck curve (derivative discontinuities occur at the vertices for a polygonal curve). Over curved two-dimensional domains, Dyken and Floater [35] assessed the approximation properties of the transfinite mean value interpolant as well as its use to solve the Poisson equation with web-splines [32], and Chin and Sukumar [84] have used it in verification tests of a cubature rule for numerical integration over curved regions. 

# •

# x

# •v•y1(x, v)

# •y2(x, v)

# •

# y3(x, v)

# c(t) 

> Figure 12: Nonconvex domain bounded by the curve c(t). The variables that appear in (15) and (16) are shown.

5. Imposing Boundary Conditions in Deep Neural Networks 

Let ˜ ubc nn (x; θ) denote the PINN trial function. We present the construction of ˜ ubc nn (x; θ) so that it exactly satisfies all essential and Robin (natural boundary condition is a particular case) boundary conditions. The trial function includes 12 00.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 (a) 00.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 (b) 00.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 

(c) 00.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 (d) 

Figure 13: Approximate distance fields on curved domains using φ(x) = 2/W1(x), with W1(x) given in (16). Surface and contour plots are shown for an (a) elliptical disk, (b) annulus, (c) hypocycloid, and (d) propeller [76]. 

the contribution from the neural network approximation, ˜ uR

nn (x; θ), where θ contains the unknown parameters of the network. We defer the presentation of ˜ uR

nn (x; θ) to Section 6. 

5.1. Normalizing functions and solution structures 

Let Ω ⊂ R2 be an open, bounded domain with boundary ∂Ω. Consider a smooth function u(x) : R2 → R and let φ(x) : R2 → R be an approximate distance function to ∂Ω that is normalized to the m-th order (see Section 2). Rvachev defined the normalizers to u of the m-th order with respect to φ(x) via the transformation [23, 25] 

u∗(x) = u(x − φ∇φ), (17) which leads to u∗(x) = u(x) on ∂Ω and ∂ku∗/∂ν k = 0 ( k = 1, 2, . . . , m) on ∂Ω . Since we are treating Dirichlet and Robin boundary conditions in this paper, we proceed to show that (17) is normalized up to order 1. If u(x) is specified on ∂Ω, then since φ = 0 on ∂Ω it implies that u∗(x) = u(x) on ∂Ω, which establishes zeroth-order normalization. The proof for first-order normalization follows. 

Proof . Let t := x − φ∇φ. Then, by the chain rule we have 

∇u∗ = ∇t u∗ · ∇ ⊗ t = ∇t u∗ · [I − φ∇ ⊗ ∇ φ − ∇ φ ⊗ ∇ φ],

where ⊗ is the dyadic (tensor) product and ∇t (·) is the gradient with respect to t. On ∂Ω, φ = 0, t = x and ∇φ = ν,since φ is normalized to the first order. Hence, we can write 

∂u∗

∂ν 

∣∣∣∣∣∂Ω

= [∇u∗ · ν]∂Ω = [∇u∗]∂Ω ·

([I − φ∇ ⊗ ν − ν ⊗ ν] · ν

)

∂Ω

= [∇u∗]∂Ω · (ν − ν) = 0,

which is the desired result. 

Note that this result also holds if we consider the unit outward normal vector, n = −ν. By extension, if φ is normalized to the m-th order, one can establish that u∗ in (17) is normalized to the m-th order, i.e., 

u∗(x) = u(x) on ∂Ω, ∂ku∗

∂ν k = 0 on ∂Ω (k = 1, 2, . . . , m). (18) Let u and its higher order normal derivatives along ν be prescribed on ∂Ω, i.e., 

u(x) = u0(x) on ∂Ω, ∂ku

∂ν k = uk(x) on ∂Ω (k = 1, 2, . . . , m). (19) 13 Then, one can represent u in the vicinity of ∂Ω and in the direction (inward) normal to the boundary using a polynomial Taylor series expansion of u in terms of φ. Rvachev et al. [23, 24] referred to this as a generalized Taylor series expansion, which takes the form: 

u(x) = u∗

> 0

(x) +

> m

∑

> k=1

u∗

> k

(x)

k! φk(x) + φm+1(x) Ψ(x), (20) where u∗

> k

(x) = uk(x − φ∇φ) and Ψ(x) is an unknown function (approximation) in the remainder term. This equation resembles univariate Taylor series expansion about x = 0, where instead of evaluating derivatives that are scalar constants, u∗

> k

(x) in (20) are scalar-valued functions. Equation (20) is the general form of the solution structure for u

that Rvachev introduced [23, 24]. If a function u ∈ Ck(Ω), and its first m derivatives vanish on ∂Ω, then the solution structure φm+1Ψ is su ffi ciently complete in the Hilbert space Hk(Ω) to approximate u and all its derivatives up to order 

k [24]. On using (20), the boundary conditions on ∂Ω that are given in (19) are exactly met. We now present the solution structure for three distinct sets of boundary conditions (Dirichlet, Neumann and Robin) that are imposed on 

∂Ω.

5.1.1. Solution structure for Dirichlet boundary condition 

If u = g is prescribed on ∂Ω, then using u∗

> 0

(x) = g(x − φ∇φ) and m = 0 in (20), we can write: 

u(x) = g(x − φ∇φ) + φ(x)Ψ(x) = g(x) − φ(x) ∂g(x)

∂ν 

∣∣∣∣∣∂Ω

+ φ(x)Ψ(x) = g(x) + φ(x) ∂g(x)

∂n

∣∣∣∣∣∂Ω

+ φ(x)Ψ(x)

= g(x) + φ(x)Ψ(x),

(21) since n = −ν on ∂Ω, and a first order linearization of g(·) is used. Therefore, the solution structure is: 

u = g + φ˜u, (22) where ˜ u is any suitable numerical approximation. So any trial function of the form given in (22) will exactly satisfy the essential boundary condition u = g on ∂Ω. In this paper, we use deep neural networks to construct ˜ u. For the homogeneous Dirichlet problem ( g = 0) using Kantorovich’s method, Babuˇ ska et al. [19] provide an a priori error estimate in the H1 norm. 

5.1.2. Solution structure for Neumann boundary condition 

Since Neumann and higher order boundary conditions for PDEs are imposed in the direction of n, the unit outward normal vector to ∂Ω, we consider solution structures that are defined with respect to n. This is a departure from the literature on the R-function method to solve PDEs, but is aligned with how boundary-value problems are posed. For first order derivatives, linearizing u∗(x) : = u(x − φ∇φ) in the neighborhood of the boundary ( φ = 0) amounts to subtracting the variation in the normal direction ν, which leads to [23] 

u∗(x) = u(x) − [φ(x)∇u(x) · ν]∂Ω + φ2(x)Ψ(x) = u(x) − [φ(x)∇φ(x) · ∇ u(x)]∂Ω + φ2(x)Ψ(x)

= (1 + φDφ 

> 1

)( u) + φ2Ψ, Dφ 

> 1

(·) : = [−∇ φ · ∇ (·)]∂Ω = ∂(·)

∂n

∣∣∣∣∣∂Ω

, (23) where ν = −n = ∇φ on ∂Ω and Dφ 

> 1

(·) is a di ff erential operator that acts in the outward normal direction to the boundary. If ∂u/∂ n = h is prescribed on ∂Ω, then using (20) and (23), we can write 

u(x) = u0

(x − φ(x)∇φ(x)) + φ(x)u1

(x − φ(x)∇φ(x)) + φ2(x)Ψ1(x)

= [1 + φDφ 

> 1

]( u0) + φu1 + φ2Ψ(x) = [1 + φDφ 

> 1

]( u0) − φh + φ2Ψ,

since ∂u/∂ν = u1 = −h on ∂Ω, and therefore the most general solution structure can be written as: 

u = [1 + φDφ 

> 1

](˜ u1) − φh + φ2 ˜u2, (24) where ˜ u1 and ˜ u2 are arbitrary approximation functions. 14 5.1.3. Solution structure for Robin boundary condition 

If the Robin boundary condition, ∂u/∂ n + cu = h, is prescribed on ∂Ω (c := c(x), h := h(x)), then following similar steps to that taken to obtain (22) and (24), we can write the most general solution structure for this boundary condition as [23]: 

u = [1 + φ(c + Dφ

> 1

)] (˜ u1) − φh + φ2 ˜u2. (25) It is readily verified that the ansatz in (25) satisfies the boundary condition ∂u/∂ n + cu = h on ∂Ω.

5.2. Imposing inhomogeneous essential boundary conditions 

Let us extend our analysis to the case when di ff erent inhomogeneous essential boundary conditions are imposed on distinct subsets of ∂Ω. Now, let the boundary ∂Ω := Γ = ∪Ni=1Γi. The inhomogeneous essential boundary condition 

u = gi is imposed on Γi (i = 1, 2, . . . , M), and on Γi (i = M + 1, M + 2, . . . , N) we assume natural boundary conditions are imposed through the potential energy functional in the variational principle. Let φi be the ADF that is associated with Γi (i = 1, 2, . . . , M), and let φ be the approximate distance field that is composed either via R-equivalence using 

φ1 ∼ φ2 . . . φ M in (10) or using the mean value potential field, W(x), given in (14) and (16). Transfinite interpolation is the generalization of scattered data interpolation to interpolation of functions over curves and surfaces. On using the singular inverse-distance based Shepard weight function [85], we can write the transfinite interpolant as [23, 25] 

g(x) =

> M

∑

> i=1

wi(x) gi(x), wi(x) = φ−μi

> i

∑Mj=1 φ−μi

> j

=

∏Mj=1; j,i φμ j

> j

∑Mk=1

∏Mj=1; j,k φμ j

> j

, (26) where the weights wi form a partition-of-unity, and (26) interpolates gi on the set Γi. In (26), μi ≥ 1 is a constant that controls the nature of interpolation that accrues. For μi = 1, the function gi is interpolated on Γi, whereas if μi = 2, both gi and ∂gi/∂ n are interpolated on Γi. Now, on using the solution structure for the Dirichlet problem given in (22) and following the work of Rvachev et al. [23, 24], we can write the trial function in the deep Ritz method as: ˜ubc nn (x; θ) = g(x) + φ(x) ˜ uR

> nn

(x; θ), (27) where φ can also be replaced by the product ∏Mi=1 φi. Since g(x) = gi(x) on Γi and φ(x) vanishes on ∪Mi=1Γi, kinematic admissibility of (27) is verified. 

5.3. Imposing inhomogeneous essential and Robin boundary conditions 

Let us consider the case that the boundary ∂Ω := Γ = Γ 1 ∪ Γ2 with Γ1 ∩ Γ2 = ∅. The boundary conditions on Γ1

and Γ2 are: 

u = g on Γ1, ∂u

∂n + cu = h on Γ2. (28) Let φ1(x) and φ2(x) be the approximate distance functions to the boundaries Γ1 and Γ2, respectively. We use the R-equivalence ( m = 1) relation in (9) to form 

φ(x) = φ1(x)φ2(x)

φ1(x) + φ2(x) , (29) which is the ADF to the boundary Γ. For the case when the boundary is partitioned into two disjoint sets, we consider two approaches to form a trial function. The first approach uses superposition of two solution structures and has a simple form. The second method, which is based on transfinite interpolation, is applicable in general when there are multiple boundaries on which essential and Robin boundary conditions are imposed. In this paper, we apply the first approach to solve a one-dimensional problem with mixed (Dirichlet and Neumann) boundary conditions, and adopt the second approach to solve a problem with Dirichlet and Robin boundary conditions in two dimensions. 

Approach I : We form solutions structures u1 and u2 such that 

u1 = g on Γ1, ∂u1

∂n + u1 = 0 on Γ2, u2 = 0 on Γ1, ∂u2

∂n + u2 = h on Γ2, (30) and therefore the desired trial function is: u(x) = u1(x) + u2(x). Now, on using (22), we know that the function g + φ1 ˜u1

satisfies the essential boundary condition on Γ1 but does not meet the Robin boundary condition on Γ2. To satisfy the Robin boundary condition on Γ2, we normalize u1 + φ˜u1 to the first order using h = 0 in (25) to obtain 

u1 = [1 + φ(c + Dφ2

> 1

)] (g + φ1 ˜u1) = [1 + φDφ2

> 1

)] (g + φ1 ˜u1) + cφg + cφ1φ˜u1, (31a) 15 where 

Dφ2 

> 1

(·) = [−∇ φ2 · ∇ (·)]Γ2 = ∂(·)

∂n

∣∣∣∣∣Γ2

(31b) is the di ff erential operator that acts in the outward normal direction on the boundary Γ2. Similarly, using (25), the minimal structure for u2 is: 

u2 = φ(φ2 ˜u2 − h). (32) Since φ = 0 on Γ and φ2 = 0 on Γ2, the conditions on u2 in (30) are satisfied. On choosing ˜ u1 = ˜u2 := ˜uR

> nn

(x; θ) in (31) and (32) and adding them up, the ansatz ˜ ubc nn (x; θ) is: ˜ubc nn (x; θ) = φ1(x) ˜ uR

> nn

(x; θ)

+ φ(x)

[{ 

φ2(x) + c(x)φ1(x)

}

˜uR

> nn

(x; θ) + Dφ2

> 1

(

φ1(x) ˜ uR

> nn

(x; θ)

)

+ Dφ2

> 1

(g(x)) + c(x)g(x) − h(x)

]

+ g(x), (33) where φ(x) is given in (29). For mixed (Dirichlet and Robin) boundary conditions, the form of the trial function in (33) appears in Rvachev and Sheiko [23]. 

Approach II : On using (22) and (25), we select the boundary functions u1 and u2 on Γ1 and Γ2, respectively, as: 

u1 = g, (34a) 

u2 = [1 + φ2

(c + Dφ2

> 1

)] (˜ u) − φ2h, (34b) with φ1φ22 ˜u as the composite remainder term. From (26), we recall the transfinite interpolant, where we now choose 

μ1 = 1 and μ2 = 2. On carrying out a few algebraic simplifications and using (34) with ˜ u := ˜uR

> nn

(x; θ), we can write the trial function ˜ ubc nn (x; θ) that exactly imposes the mixed boundary conditions as: ˜ubc nn (x; θ) = w1(x)u2(x; θ) + w2(x)u1(x) + φ1(x)φ22(x)˜ uR

> nn

(x; θ), (35a) where 

w1(x) = φ1(x)

φ1(x) + φ22(x) , w2(x) = φ22(x)

φ1(x) + φ22(x) , (35b) and 

u1(x) = g(x), u2(x; θ) = [1 + φ2(x)(c(x) + Dφ2

> 1

)] ( ˜uR

> nn

(x; θ)) − φ2(x)h(x). (35c) 

6. Approximation of Trial Functions in a Deep Neural Network 

In this paper, we exclusively use the densely connected neural network architecture, also known as the multi-layer perceptron (MLP), which has its origin in the early works of Rosenblatt [86]. MLPs consist of multiple layers of neurons, where each neuron has the task of converting its input to an output by generally passing it through a nonlinear function called activation. MLPs are characterized by an architecture where neurons in a given layer are connected densely to the neurons in the neighboring layers (Fig. 14). We note in passing, however, that the latest revolution in deep learning began with a di ff erent neural network architecture—convolutional neural networks (CNN)—applied to image classification tasks [87]. Independent of research in PINNs, some of these modern architectures have also been applied to mechanics problems [88]. In keeping with the broadly accepted definition, we consider any deep network to have two or more hidden layers. In a standard collocation or Ritz method, the trial function is expanded as a linear combination of known basis functions. The point of departure in using deep neural networks is that the ansatz herein is represented by a nonlinear map that consists of unknown parameters. These parameters are obtained via the solution of a minimization problem, which in general but not necessarily, is a least squares optimization problem. Once the parameters are determined, one obtains the numerical solution to the boundary-value problem. 16 6.1. Feedforward neural network 

Given a point x ∈ Rd, we use a multilayer feedforward deep neural network to construct ˜ uR

> nn

(x; θ), which is then used to build ˜ ubc nn (x), the approximation to u(x) : Rd → R. The layers in between the input and output layer are known as hidden layers. Each hidden layer consists of neurons (hidden units), and each neuron in a hidden layer takes its input from the neurons in the preceding layer and computes its own activation. A network diagram of neural networks with one, two, and four hidden layers is shown in Fig. 14. In this paper, we consider the following boundary-value problem: 

Lu(x) = f (x) in Ω ⊂ Rd, (36a) 

Buu(x) = g(x) on Γu, (36b) 

Bnu(x) = h(x) on Γn, (36c) where L is in general a di ff erential operator plus the identity and d is the spatial dimension. In (36), Ω is an open bounded domain with boundary ∂Ω = Γ u ∪Γn and Γu ∩Γn = ∅. Equation (36b) represents essential boundary conditions for second- and fourth-order problems, and (36c) imposes natural and higher order boundary conditions. For a second-order problem, (36c) is a Robin boundary condition. As noted in Section 5, we form the trial function to approximate 

u(x) as a combination of terms that involve the approximate distance functions (φk(x), φ(x)) and the neural network approximation. Consider a neural network that consists of L hidden layers with N` neurons in the hidden layer `

and activation function σ : R → R. The size of the neural network is: N = ∑L 

> `=1

N`. Let ˜ uR

> nn

(x; θ) be the PINN approximation, where θ := {W, b} is the unknown parameter vector, with weights W` ∈ RN` ×N `−1 and biases b` ∈ RN` .We write ˜ uR

> nn

(x; θ) via the composition of T (`) (` = 1, 2, . . . , L) and a linear map G as: ˜uR

> nn

(x; θ) = G ◦ T (L) ◦ T (L−1) ◦ . . . ◦ T (1) (x), (37) where G : RNL → R is the linear mapping for the output layer and in each hidden layer ( ` = 1, 2, . . . , L), the nonlinear mapping is: 

T (`)(z) = σ(W` · z + b`), (38) where z ∈ RN`−1 . For a neural network with activation function σ and a single hidden layer that consists of N neurons, the PINN approximation is: ˜uR

> nn

(x; θ) =

> N

∑

> i=1

ci σ(wi · x + bi), (39) where wi ∈ Rd and bi, ci ∈ R. It is known that a multilayer neural network is su ffi ciently rich to be able to approximate any L2 function to arbitrary accuracy [66, 67, 89]. However, realization of this in practice hinges on the width and depth of the network, choice of the activation function and the computational algorithm to solve the optimization problem. 

6.2. Backpropagation algorithm 

Determination of the optimal parameters of the network is done through an iterative optimization process called backpropagation and a popular algorithm for backpropagation is stochastic gradient descent, which is the stochastic version of the gradient descent algorithm. An important step in this procedure is the e ffi cient computation of the gra-dient of the loss function using automatic di ff erentiation. In this paper, we exclusively use the Adam backpropagation algorithm whose details are given in [91]. Adam is an extension of the stochastic gradient descent algorithm and di ff ers from it primarily in its implementation of per-parameter learning rates that are continuously tweaked during learning through both the first and second moments of the gradients. 

7. Formulations 

We now present the formulations for deep collocation and deep Ritz for second- and fourth-order problems. In the collocation approach, all boundary conditions are exactly satisfied; for the Ritz method, the essential boundary conditions are met. 17 (a)     

> (b)
> (c)
> Figure 14: Deep neural network with (a) one, (b) two, and (c) four hidden layers [90]. Input layer is a point x∈R2, and the output layer is the PINN approximation, ˜ uR
> nn (x;θ).

7.1. Deep collocation 

Let us consider a second-order boundary-value problem with mixed boundary conditions. We require the trial function given in (35) (meets all boundary conditions) to also satisfy the governing equation in (36a) at NI interior collocation points. We label these collocation points as {xk}NI

> k=1

. When substituted in (36a), this defines a residual error at each point xk and to determine the parameter θ, we minimize the mean squared residual error: 

θ∗ = arg min 

> θ

Lbc nn (θ), Lbc nn (θ) : = ||L ˜ubc nn (x; θ) − f (x)|| 2 

> Ω,NI

= 1

NINI∑

> k=1

[L˜ubc nn (xk; θ) − f (xk)]2

, (40) where Lbc nn (θ) is known as the loss function, and || · || Ω,NI denotes the mean discrete L2 norm of its argument over the domain Ω that is discretized using NI collocation points. The Adam algorithm [91] is used to solve (40). When a standard PINN trial function, ˜ unn (x; θ), is used, additional residual error contributions from the boundary conditions are present in the loss function. If only essential boundary conditions are imposed, with u = g on ∂Ω, then the solution for the parameter θ is given by 

θ∗ = arg min 

> θ

Lnn (θ), Lnn (θ) : = ||L ˜unn (x; θ) − f (x)|| 2 

> Ω,NI

+ || ˜unn (x; θ) − g(x)|| 2 

> ∂Ω,NB

, (41) where ||·|| ∂Ω,NB is the mean discrete L2 norm of its argument over the boundary ∂Ω, and NB is the number of collocation points on ∂Ω. As noted in Section 1, the first and second terms in Lnn (θ) are referred to as the interior (PDE) loss and the boundary loss, respectively. 

7.2. Deep Ritz 

E and Yu [9] introduced the deep Ritz method to solve low- and high-order boundary-value problems that have a variational structure. Samaniego et al. [92] have applied the Ritz approach to solve problems in computational solid mechanics. We consider second-order (Poisson) and fourth-order (plate bending) boundary-value problems. Essential and mixed boundary conditions are considered for the Poisson problem and clamped boundary conditions are imposed for the plate bending problem. We use a variational principle (minimization of the potential energy functional) to solve both problems. A trial function, ˜ ubc nn (x; θ), from a finite-dimensional space is used in the variational principle, which forms the basis of the deep Ritz method. 

7.2.1. Second-order problems 

Referring to the model boundary-value problem in (36), we first consider a Poisson problem with Dirichlet bound-ary conditions: 

−∇ 2u = f in Ω, (42a) 

u = g on ∂Ω. (42b) 18 The variational principle for this problem is: min 

> u∈S

[

Π[u] : = 12 a(u, u)

︸ ︷︷ ︸

> Wint [u]

− `(u)

︸︷︷︸ 

> Wext [u]

, S =

{

u : u ∈ H1(Ω), u = g on ∂Ω

} ] 

, (43a) where 

a(u, w) =

∫

> Ω

∇u · ∇ w d x, `(w) =

∫

> Ω

f wd x (43b) are a symmetric bilinear functional and a linear functional, respectively, and u and w are the trial and test functions, respectively. In (43a), Wint [u] is the internal work (strain energy) and Wext [u] is the external work, and Hk(Ω) is the Sobolev space that consists of functions that have square integrable derivatives up to order k in Ω.As a second problem, we consider a Poisson problem with mixed (Dirichlet and Robin) boundary conditions: 

−∇ 2u = f in Ω, (44a) 

u = g on Γu, ∂u

∂n + cu = h on Γn, (44b) where n is the unit outward normal on Γn, c := c(x) and h := h(x) are boundary data, and Γ = ∂Ω with Γ = Γ u ∪ Γn

and Γu ∩ Γn = ∅. The variational principle for this problem is: min 

> u∈S

[

Π[u] : = 12 a(u, u) − `(u), S =

{

u : u ∈ H1(Ω), u = g on ∂Ω

} ] 

, (45a) where 

a(u, w) =

∫

> Ω

∇u · ∇ w dx +

∫

> Γn

cuw dS , `(w) =

∫

> Ω

f w d x +

∫

> Γn

hw dS . (45b) As indicated in (27), we choose a kinematically admissible trial function so that the essential boundary condition is satisfied. On substituting the PINN trial function in (43) or (45), the finite-dimensional minimization problem becomes min 

> θ

[

Π[˜ ubc nn ] : = 12 a (˜ubc nn (x; θ), ˜ubc nn (x; θ)) − ` (˜ubc nn (x; θ))]

. (46) To compute the potential energy functional, we use a Monte Carlo integration rule with points that are distributed (randomly or quasi-uniformly) in the domain with a constant weight that is attached to each point. To obtain the unknown parameters, we solve a discrete nonlinear minimization problem that is posed as: 

θ∗ = arg min 

> θ

Lbc nn (θ), Lbc nn (θ) = 1

NINI∑

> k=1

[ 12 a (˜ubc nn (xk; θ), ˜ubc nn (xk; θ)) − ` (˜ubc nn (xk; θ))]

, (47) where Lbc nn (θ) is the loss function, and a(·, ·) and `(·) are given in (43b) and (45b) for the Poisson problems with Dirichlet and mixed boundary conditions, respectively. In general, the stochastic gradient descent algorithm or a variant of it is used to solve (47). 

7.2.2. Fourth-order problem 

We consider the fourth-order problem of Kirchho ff plate bending with clamped boundary conditions. The strong form of the boundary-value problem is: 

∇4u = f in Ω ⊂ R2, (48a) 

u = 0 on ∂Ω, u,n := ∂u

∂n = 0 on ∂Ω, (48b) where u := u(x, y) is the out-of-plane plate deflection, f is the transverse load per unit area, and n is the unit outward normal on the boundary ∂Ω. The flexural rigidity of the plate is assumed to be unity. 19 The variational principle that is associated with the strong form in (48) is: min 

> u∈S

[

Π[u] : = 12 a(u, u) − `(u), S =

{

u : u ∈ H2(Ω), u = 0 on ∂Ω, u,n = 0 on ∂Ω

}] 

, (49a) where 

a(u, w) =

∫

> Ω

(∇2u)( ∇2w) dx, `(w) =

∫

> Ω

f w d x, (49b) and now both essential boundary conditions in (48b) must be met to satisfy kinematic admissibility. To meet this objective, the solution structure for the PINN trial function is chosen as: ˜ubc nn (x; θ) = [φ(x)] 2 ˜uR

> nn

(x; θ), (50) where φ(x) is an approximate distance function to the boundary ∂Ω.On substituting the trial function in (49), the finite-dimensional minimization problem is of the same form as (43), where a(·, ·) and `(·) are now given by (49b). Similar to the deep Ritz approach for the Poisson problem, we use a Monte Carlo integration rule with equally-weighted points. Following the same steps, the loss function is of the same form as in (47) with the bilinear and linear functionals defined in (49b). 

8. Numerical Implementation 

Numerical solutions using deep neural networks are presented for one-, two- and four-dimensional boundary-value problems. Both, second- and fourth-order problems are considered. The trial function, ˜ ubc nn (x; θ), contains ˜ uR

> nn

(x; θ), the approximation that is composed by the neural network. The trial function in the standard PINN [6] is denoted by ˜unn (x; θ). Likewise, Lbc nn (θ) is used to denote the loss function in our approach and Lnn (θ) is that for standard PINN. We refer to the numerical solution that is obtained (after training) by our approach as ˜ ubc nn (x) and that obtained (after training) using standard PINN [6] as ˜ unn (x). Deep collocation and deep Ritz methods are used to solve boundary-value problems. For the Ritz method, either ReLU or cubic ReLU activation function is used, with the problem that is solved in Section 9.1.7 being the sole exception where the Gaussian activation function is used. Unless otherwise stated, R-equivalence (REQ) composition with m = 1 and mean value potential (MVP) with p = 1 are used to form approximate distance functions in ˜ ubc nn (x; θ). For all problems that are considered in this paper, whenever REQ is indicated, we employ (10) to form φ(x); when MVP is mentioned, the expression for φ given in (14) or (16) applies. For collocation, all boundary conditions are exactly satisfied, and essential boundary conditions are exactly met in the Ritz method, which ensures that the trial function is kinematically admissible. All collocation points are considered in a single batch in the network; points are not sorted in bins and passed in batches, which in general is a more e ffi cient approach. The formulation described in Section 7 has been implemented using Google’s JAX library in Python [93], which can automatically di ff erentiate native Python and NumPy functions. As an example, the code listing below sets up the calculation for the approximate distance function over an arbitrary polygon through the R-equivalence operation in (10). 

def dist(x1,y1,x2,y2): 

return jax.numpy.sqrt((x2-x1)**2+(y2-y1)**2) 

def linseg(x,y,x1,y1,x2,y2): L = dist(x1,y1,x2,y2) xc = (x1+x2)/2. yc = (y1+y2)/2. f = (1/L)*((x-x1)*(y2-y1) - (y-y1)*(x2-x1)) t = (1/L)*((L/2.)**2-dist(x,y,xc,yc)**2) varphi = jax.numpy.sqrt(t**2+f**4) phi = jax.numpy.sqrt(f**2 + (1/4.)*(varphi - t)**2) 

return phi 

def phi(x,y,segments): 

20 m = 1. R = 0. for i in range(len(segments[:,0])): phi = linseg(x,y,segments[i,0],segments[i,1],segments[i,2],segments[i,3]) R = R + 1./phi**m R = 1/R**(1/m) 

return R

Here segments is a NumPy array, which contains the coordinates of the line segments that make up the polygon and phi(x,y,segments) returns an approximate distance function from the location (x,y) to the polygon. A multilayer perceptron neural network is created using the following set of functions: 

def RePU3(x): 

return (jax.numpy.maximum(0, x**3)) 

def repu3_layer(params, x): 

return RePU3(jax.numpy.dot(params[0], x) + params[1]) 

def NN(params, x, y): """ Compute the forward pass for each example individually """ activations = jax.numpy.array([x,y]) # Loop over the RePU3 hidden layers for w, b in params[:-1]: activations = repu3_layer([w, b], activations) final_w, final_b = params[-1] final = jax.numpy.sum(jax.numpy.dot(final_w, activations)) + final_b 

return (final[0]) 

In the above, the cubic ReLU function is used as the activation in the hidden layers, which is also an instance of the Rectified Power Unit (RePU) activation function RePU n(x) : = [max(0 , x)] n (n = 3 for cubic ReLU). A linear activation function is used in the output layer. Here, params is a NumPy array consisting of the optimizable parameters, (w,b) ,of the network, and its shape is determined by the architecture of the network. A network architecture that has been used frequently in this paper for problems in Rd is a 2 hidden layer network d–N–N–1, where N is the number of neurons in each hidden layer and is typically 50. Finally, the following functions and their derivatives are used to construct the ansatz ˜ ubc nn (x; θ), which satisfies homogeneous essential boundary conditions that are imposed on the boundary of a polygonal domain: 

def u(params, x, y): 

return phi(x,y,segments)*NN(params,x,y) #Examples of first-order partial derivatives gradx = grad(u,1) grady = grad(u,2) #Examples of second-order partial derivatives gradxx = grad(grad(u,1),1) gradyy = grad(grad(u,2),2) gradxy = grad(grad(u,1),2) 

Once the derivatives are formed, the appropriate loss functions are constructed in the interior of the domain to solve the problem through Ritz or collocation. For all problems that are solved in this paper, network training is done using Google’s Colaboratory cloud platform [94]. Single-precision arithmetic is used in the computations. An important consideration is the generation of collocation points in the interior and on the boundary of the domain to evaluate the loss terms. When the domain is simple—for example, a square or a hypercube—then these points are 21 −1 0 1      

> −1.0
> −0.50.00.51.0
> −101
> −1.0
> −0.50.00.51.0
> 0.00.51.00.00.20.40.60.81.0Figure 15: Representative meshes generated from dmsh and the corresponding collocation points.

generated on a uniform grid in Rd. This is the case for the solution of the heat equation on square domains, Eikonal problems in Section 10.5, and for the Poisson problem over the 4-dimensional hypercube in Section 11. In other examples that involve more complicated domains, we use the Python library dmsh [95], which draws inspiration from distmesh [96], to create triangular meshes and we use the centroids of the generated triangles as the interior collocation points. Figure 15 shows a few representative meshes generated by dmsh that are used in this paper. The corresponding interior collocations points are shown as dots. When needed, dmsh is also used to create collocation points on the boundaries of the domain. 

9. Numerical Examples in One Dimension 

We consider several second-order problems that involve essential and mixed boundary conditions. As a prototyp-ical fourth-order problem, we solve the deflection for a clamped Euler-Bernoulli beam, and also investigate a single hidden layer meshfree RBF-network solution in Section 9.1.7. Our objective is to demonstrate the benefits of the new formulation vis-` a-vis standard PINN [6] in which equal weights are chosen for the PDE and boundary loss terms. Hence, our emphasize is not on obtaining the most accurate PINN solution by finding the optimal hyperparameters nor comparing its performance versus the finite element method for forward problems. We compare our results with those obtained using standard PINN [6] and to the exact solution. 

9.1. Deformation of a homogeneous elastic rod 

Consider the boundary-value problem for the deformation of an elastic rod (Youngs’s modulus and cross-sectional area are taken as unity): 

u′′ + b = 0 in Ω = (x1, x2) (51a) 

u(x1) = g, u′(x2) + cu (x2) = h, (51b) where u := u(x) is the axial displacement field, u′(x) is the strain field, b := b(x) is the axial body force per unit length, and c, g and h are constants. The second boundary condition is a Robin boundary condition; if the bar is connected to a spring that is attached to a fixed end, then h = 0. We select test problems with di ff erent boundary conditions, and vary the regularity of b(x) from it being a smooth function to a δ-function, and even choose b(x) that has a singularity at the origin. 

9.1.1. Example 1 

As the first example, a Dirichlet problem in Ω = (0 , 1) is selected with body force b(x) = 1 − 2x + 10 x2, and boundary conditions u(0) = 1/2 and u(1) = −1/2. The exact solution is: u(x) = 1/2 − x2/2 + x3/3 − 10 x4/12. The exact signed distance functions to x = 0 and x = 1 are φ1(x) = x and φ2(x) = 1 − x, respectively. Now, we join these to form a smooth approximate distance function to the boundary ∂Ω = {0, 1}. On using the product of φ1 and φ2, and 22 the R-conjunction ( α = 0) and R-equivalence ( m = 2) relations in (2) and (9), respectively, we obtain the following combined distance functions: 

φA(x) = φ1(x)φ2(x), φB(x) = φ1(x) + φ2(x) −

√

φ21(x) + φ22(x), φC (x) = φ1(x)φ2(x)

√

φ21(x) + φ22(x)

. (52) Note that in one dimension, the product formula is normalized to order 1, but this does not generalize to higher dimensions. Coincidentally, for any domain Ω = (x1, x2) in one dimension, the product formula scaled by L := x2 − x1

coincides with m = 1 in the R-equivalence relation. In the numerical computations, the trial function is formed using (22): ˜ubc nn (x; θ) = g(x) + φ(x) ˜ uR

> nn

(x; θ), g(x) = 1 − 2x

2 , (53) where g(x) is formed using the transfinite interpolant in (26), ˜ uR

> nn

(x; θ) is the neural network approximation, and φ(x)is chosen to be either φA(x), φB(x) or φC (x). Note that we did not solve the patch test ( b = 0) since the exact solution, 

u(x) = g(x), is already captured by the presence of g(x) in (53). The network architecture 1–30–30–1 is used. We compute collocation solutions using φA, φB and φC in (53), and also for the standard PINN approximation, ˜ unn (x; θ) [6]. In Fig. 16, the numerical results are presented. All approaches are able to reach losses of the same order, and there is a good match between ˜ ubc nn (x) and the exact displacement field. From Fig. 16d, we observe that the errors in the displacement and strain fields using ˜ ubc nn are uniformly smaller than those obtained using ˜ unn . This di ff erence stems from the fact that in our approach the trial function is constructed with the exact satisfaction of the boundary conditions. 0 2000 4000 Epochs 10 −2         

> 10 −1
> 10 0
> 10 1
> Training loss
> φA
> φB
> φC
> ˜unn
> 02000 4000 Epochs 10 −3
> 10 −2
> 10 −1
> 10 0
> Error
> φA
> φB
> φC
> ˜unn
> 0.00 0.25 0.50 0.75 1.00
> x
> −0.6
> −0.4
> −0.20.00.20.40.6
> u(x), ˜ubc nn (x)
> exact ˜ubc nn (φB)0.00.51.0
> x
> −0.02
> −0.01 0.00 0.01 0.02 Prediction errors
> u- ˜ ubc nn
> u′- ˜ ubc nn ′
> u- ˜ unn
> u′- ˜ unn ′

(a) (b) (c) (d)                    

> Figure 16: Collocation solutions for a Dirichlet problem. The body force b(x)=1−2x+10 x2, and the essential boundary conditions are u(0) =1/2and u(1) =−1/2. The network architecture is 1–30–30–1. Numerical solutions, ˜ ubc nn (x), are computed using di ff erent ADFs ( φA,φB,φC), and are compared to the solution obtained using standard PINN, ˜ unn (x). (a), (b) Training loss and normalized absolute error in the displacement field as a function of epochs. (c) Exact solution u(x) and ˜ ubc nn (x) (using φB) are shown, and (d) Errors in the displacement and strain fields are compared.

We emphasize that it is possible for both standard PINN and our formulation to deliver better accuracy if a larger network, more interior collocation points, and network training for a longer duration are chosen. This is realized at the expense of more computing time. So for just this example, we demonstrate the same. We now use a 1–50–50–1 network architecture with 300 interior collocation points. The training is conducted until 50,000 epochs. The solutions are presented in Fig. 17, which reveal that both approaches are now much more accurate than the results that are shown in Fig. 16. From Fig. 17, we observe that the solution obtained using standard PINN and our approach have relative errors of O(10 −5) and O(10 −6), respectively. So even at smaller relative errors, we note that our approach is more accurate than standard PINN for the same values of the hyperparameters. 

9.1.2. Example 2 

We reconsider the problem posed in Example 1, with Dirichlet boundary condition at x = 0 but homogeneous Neumann (traction-free) condition at x = 1. For the domain Ω = (0 , 1), body force b(x) = 1 − 2x + 10 x2, and boundary conditions u(0) = 1/2 and u′(1) = 0, the exact solution is: u(x) = 1/2 + 10 x/3 − x2/2 + x3/3 − 10 x4/12. The trial function, ˜ ubc nn (x; θ), is formed using (33), with φ1(x) = x, φ2(x) = 1 − x, φ(x) = x(1 − x), g = 1/2, and c = h = 0: ˜ubc nn (x; θ) = x ˜uR

> nn

(x; θ) + x(1 − x) [(1 − x) ˜ uR

> nn

(x; θ) + {x ˜uR

> nn

(x; θ)}′

> x=1

] + 12 . (54) 23 0 20000 40000 Epochs 10 −6         

> 10 −4
> 10 −2
> 10 0
> Training loss
> φA
> φB
> φC
> ˜unn
> 020000 40000 Epochs 10 −5
> 10 −3
> 10 −1
> Error
> φA
> φB
> φC
> ˜unn
> 0.00 0.25 0.50 0.75 1.00
> x
> −0.6
> −0.4
> −0.20.00.20.40.6
> u(x), ˜ubc nn (x)
> exact ˜ubc nn (φB)0.00.51.0
> x
> −2
> −10Prediction errors
> ×10 −5
> u- ˜ ubc nn
> u- ˜ unn

(a) (b) (c) (d) 

Figure 17: Collocation solutions for the Dirichlet problem that is presented in Example 1. The network architecture is 1–50–50–1 with 300 interior collocation points. The captions for (a), (b), (c), (d) mirror those shown in Fig. 16. 

The network architecture 1–50–50–1 is used, since the network 1–30–30–1 did not converge for standard PINN. The collocation solutions ˜ ubc nn (x) and ˜ unn (x) are compared to the exact solution u(x) in Fig. 18. We observe from Fig. 18a that the training loss for ˜ ubc nn (x; θ) is about two orders smaller than ˜ unn (x; θ), whereas in Fig. 18b, the corresponding normalized absolute error is one order smaller. The numerical solution ˜ ubc nn (x) is in excellent agreement with the exact solution in Fig. 18c. Over the entire interval x ∈ [0 , 1], we find that the displacement and strain fields from ˜ ubc nn (x) are markedly more accurate than those from ˜ unn (x) (see Fig. 18d). 0 2500 5000 7500 Epochs 10 −3       

> 10 −1
> 10 1
> Training loss ˜ubc nn
> ˜unn
> 02500 5000 7500 Epochs 10 −4
> 10 −3
> 10 −2
> 10 −1
> 10 0
> Error ˜ubc nn
> ˜unn
> 0.00.51.0
> x
> 0.51.01.52.02.5
> u(x), ˜ubc nn (x)
> exact ˜ubc nn
> 0.00.51.0
> x
> −0.0075
> −0.0050
> −0.0025 0.0000 0.0025 Prediction errors
> u- ˜ ubc nn
> u′- ˜ ubc nn ′
> u- ˜ unn
> u′- ˜ unn ′

(a) (b) (c) (d) 

Figure 18: Collocation solutions for a Neumann problem. The body force b(x) = 1 − 2x + 10 x2, and u(0) = 1/2 and u′(1) = 0. The network architecture is 1–50–50–1. Numerical solution using ˜ ubc nn (x; θ) and ˜ unn (x; θ) are compared. (a), (b) Training loss and normalized absolute error as a function of epochs. (c) ˜ ubc nn (x) is compared to the exact solution. (d) Errors in the displacement and strain fields for ˜ ubc nn (x) and ˜ unn (x). 

9.1.3. Example 3 

For this example, we choose Ω = (0 , 1), a sinusoidal body force b(x) = − sin( kπx) with varying k, and essential boundary conditions u(−1) = u(1) = 0. The exact solution is: u(x) = − sin( kπx)/π 2k2. This example serves to reveal the spectral (low-frequency) bias [97, 98] of neural network approximations. In Fig. 19, the numerical results are presented. For k = 1 and using standard PINN, we found that the normalized absolute error for the networks 1–30–30–1 and even 1–100–100–1 did not converge; it took a network architecture of 1–100-100-100-1 for the error to be comparable to that obtained using ˜ ubc nn (x; θ) on a 1–30–30–1 architecture. The normalized absolute errors as a function of epochs is shown in Fig. 19a. The numerical solutions are compared to the exact solution u(x) in Fig. 19b, and we notice the poor solution that is generated by ˜ unn (x; θ) on the 1–30–30–1 network. For k = 3, 5 on a 1–100– 100-1 network, the normalized absolute error in the ˜ ubc nn (x) displacement field is shown in Fig. 19c, and ˜ ubc nn (x) and 

u(x) are compared in Fig. 19d for k = 5. Good agreement between ˜ ubc nn (x) and u(x) is realized. We attribute the poor performance of the standard PINN approach to the fact that the boundary conditions are poorly approximated, whereas with ˜ ubc nn (x; θ) the boundary conditions are exactly satisfied. This observation is in broad agreement with the findings of Wang et al. [16], who further analyze the source of this discrepancy by drawing attention to the contributions from the boundary and interior terms in the loss function. For k ≥ 10 with ˜ ubc nn (x; θ), accurate PINN solutions using a single neural network for the entire domain becomes infeasible due to the spectral bias of the neural network approximation; one can adopt a domain-decomposition strategy to obtain accurate numerical solutions for high-frequency problems. 24 0 1000 2000 3000 Epochs 10 −3                    

> 10 −2
> 10 −1
> 10 0
> Error ˜ubc nn
> ˜unn −1˜unn −2˜unn −3
> −101
> x
> −2
> −1012
> u(x), uh(x)
> exact ˜ubc nn
> ˜unn −1˜unn −3020000 40000 Epochs 10 −5
> 10 −4
> 10 −3
> 10 −2
> Error
> k=3
> k=5
> −101
> x
> −0.004
> −0.002 0.000 0.002 0.004
> u(x), ˜ubc nn (x)
> exact ˜ubc nn

(a) k = 1 (b) (c) (d) k = 5

Figure 19: Collocation solutions for a homogeneous Dirichlet problem with a sinusoidal body force b(x) = − sin( kπx) with varying k. (a) Normalized absolute error as a function of epochs for k = 1. ˜ ubc nn (x; θ) on a 1–30–30–1 architecture is compared to ˜ unn (x; θ). ˜ unn –1, ˜ unn –2 and ˜ unn –3 are solutions that are obtained on network architectures 1–30–30–1, 1–100–100–1 and 1–100–100–100–1, respectively. (b) ˜ unn (x), ˜ ubc nn (x), and u(x) are plotted for k = 1. (c) Normalized absolute error during training using ˜ ubc nn (x; θ) for k = 3 and k = 5. Network architecture is 1–100–100–1. (d) For k = 5, error in the displacement field for ˜ unn (x). 

9.1.4. Example 4 

Consider an elastic rod that occupies Ω = (−1, 1) and is subjected to a discontinuous body force b(x) = H(x), where 

H(x) is the Heaviside function. Essential boundary conditions are imposed at both ends: u(−1) = 0 and u(1) = −1/2. The exact solution, u(x) ∈ C1(Ω), is: 

u(x) =



0 , −1 ≤ x < 0

− x2 

> 2

, 0 ≤ x ≤ 1 . (55) Numerical computations are performed on a 1–50–50–1 network. Numerical results for ˜ unn and ˜ ubc nn are presented in Fig. 20. The training loss of ˜ ubc nn (x; θ) at 10,000 epochs is O(10 −3) but ˜ unn (x; θ) stagnates to a value just below 0 .1. These losses correspond to normalized absolute errors of O(10 −5) and O(10 −4) at the end of the training for ˜ ubc nn (x) and ˜unn (x), respectively (see Fig. 20b). Figures 20c and 20d reveal that ˜ ubc nn (x) is in good agreement with u(x), whereas the errors in ˜ unn (x) are significant, and are especially pronounced in the vicinity of x = 0. The PDE loss is dominant over the boundary losses. We point out that if both boundary conditions are homogeneous then the accuracy of ˜ unn (x) is comparable to that of ˜ ubc nn (x). 0 5000 10000 Epochs 10 −3      

> 10 −2
> 10 −1
> Training loss ˜ubc nn
> ˜unn
> 05000 10000 Epochs 10 −4
> 10 −3
> 10 −2
> 10 −1
> Error ˜ubc nn
> ˜unn
> −101
> x
> −0.5
> −0.4
> −0.3
> −0.2
> −0.10.0
> u(x), uh(x)
> exact ˜ubc nn
> ˜unn
> −101
> x
> −0.10
> −0.05 0.00 0.05 Prediction errors
> u- ˜ ubc nn
> u′- ˜ ubc nn ′
> u- ˜ unn
> u′- ˜ unn ′

(a) (b) (c) (d) 

Figure 20: Collocation solutions for a Dirichlet problem with a discontinuous body force, b(x) = H(x). Essential boundary conditions are u(−1) = 0and u(1) = −1/2, and the network architecture is 1–50–50–1. (a), (b) Training loss and normalized absolute error as a function of epochs for ˜ubc nn (x; θ) and ˜ unn (x; θ). (c) Comparisons of ˜ ubc nn (x) and ˜ unn (x) with the exact solution. (d) Error in the displacement and strain fields. 

9.1.5. Example 5 

Let us consider an elastic rod that occupies Ω = (−1, 1) and is subjected to a unit point load at the origin, i.e., 

b(x) = δ(x), where δ(x) is the δ-function. Homogeneous essential boundary condition is prescribed at x = −1 and traction-free conditions prevail at x = 1. The exact solution, u(x) ∈ C0(Ω), is: 

u(x) =



1 + x , −1 ≤ x < 01 , 0 ≤ x ≤ 1 , (56) 25 which has a kink at x = 0. We use the deep Ritz method with ˜ ubc nn (x; θ) to solve this problem since it is not possible to solve this problem using the collocation method ( δ(x) is a distribution that is defined over a zero measure). The network architecture is 1–50–50–1. Numerical results for ˜ ubc nn are presented in Fig. 21. From Fig. 21a, we observe that the training loss converges to a value close to −0.5 in a few epochs; this corresponds to a small normalized absolute error of O(10 −3) (see Fig. 21b). On using u given in (56), we find that the potential energy of the exact solution is: 

Π[u] = 12

∫ 1

> −1

u′2 dx − u(0) = − 12 ,

which is the target loss that Π[˜ ubc nn ] seeks to attain. Figure 21c shows excellent agreement between ˜ ubc nn (x) and the exact solution. The errors in the displacement and strain fields are presented in Fig. 21d. The errors in u are uniformly small for all x ∈ [0 , 1]. The errors in the strain field follow the same trends, but have larger errors in the vicinity of the origin. This is not surprising since ˜ ubc nn (x) is C2(Ω) (cubic ReLU activation function), whereas the exact solution u ∈ C0(Ω), with u′(x) being discontinuous at x = 0. This is also the source for the small discrepancy in the Ritz energy loss. 0 2500 5000 7500 Epochs        

> −0.5
> −0.4
> −0.3
> −0.2
> −0.10.0Training loss 02500 5000 7500 Epochs 10 −3
> 10 −2
> 10 −1
> 10 0
> Error
> −101
> x
> 0.00.20.40.60.81.0
> u(x)
> exact ˜ubc nn
> −101
> x
> −0.3
> −0.2
> −0.10.00.10.2Prediction errors
> u- ˜ ubc nn
> u′- ˜ ubc nn ′

(a) (b) (c) (d)          

> Figure 21: Ritz solution for a point load, b(x)=δ(x). Homogeneous essential boundary condition is imposed at x=−1 and traction-free conditions prevail at x=1. The network architecture is 1–50–50–1. (a), (b) Training (Ritz) loss and normalized absolute error as a function of epochs. (c) Comparison of ˜ ubc nn (x) with the exact solution. (d) Errors in displacement and strain fields.

9.1.6. Example 6 

To obtain a weakly singular solution for the elastic rod problem, we consider a body force that has a singularity at the origin. We choose Ω = (0 , 1) with b(x) = 2x−4/3/9 with essential boundary conditions u(0) = 0 and u(1) = 1. The exact solution is u(x) = x2/3, and u ∈ H1(Ω) is weakly singular. This problem is solved using the collocation approach on a 1–50–50–1 network architecture. Numerical results for ˜ ubc nn (x) and ˜ unn (x) are presented in Fig. 22. The training loss of ˜ ubc nn (x; θ) at 10,000 epochs is close to O(10 −4) and ˜ unn (x; θ) is more than two orders larger. This same trend is observed in the normalized absolute error as a function of epochs (see Fig. 22b). Figures 22c and 22d show that ˜ ubc nn (x)is in fairly good agreement with u(x), whereas the error in the displacement and strain fields of ˜ unn (x) are appreciable. It appears that ˜ unn (x) and u(x) di ff er by close to an a ffi ne function, which one can infer as being present within ˜ unn (x)to meet the essential boundary conditions. 

9.1.7. Example 7 

To draw connections to meshfree methods based on RBFs [46] and local maximum-entropy approximants [48], which are discussed in Section 1, we solve (51) with inhomogeneous Dirichlet boundary conditions. The domain 

Ω = (0 , 1), and we choose the exact solution as: 

u(x) =

> 2

∑

> i=1

exp [−γi(x − ai)2] , (57) which is a sum of two Gaussian functions, and γi and ai (i = 1, 2) are constants. The body force b(x) = −u′′ (x). Essential boundary conditions are imposed at x = 0 and x = 1 that are consistent with the exact solution in (57). In the numerical computations, we choose a1 = 1/4, a2 = 6/10, γ1 = 9 and γ2 = 10. The network architecture is 1–10–1 (1 hidden layer). For the hidden layer, we select a Gaussian activation function, σ(x) = exp( −x2), and a linear activation function for the output layer. The centers of the Gaussian are chosen to be fixed, and only the support-widths 26 0 5000 10000 Epochs 10 −3      

> 10 −1
> 10 1
> Training loss ˜ubc nn
> ˜unn
> 05000 10000 Epochs 10 −2
> 10 −1
> 10 0
> Error ˜ubc nn
> ˜unn
> 0.00.51.0
> x
> 0.00.20.40.60.81.0
> u(x), uh(x)
> exact ˜ubc nn
> ˜unn
> 0.00.51.0
> x
> −0.2
> −0.10.00.1Prediction errors
> u- ˜ ubc nn
> u′- ˜ ubc nn ′
> u- ˜ unn
> u′- ˜ unn ′

(a) (b) (c) (d) 

Figure 22: Collocation solutions for a Dirichlet problem with a singular body force, b(x) = 2x−4/3/9. Essential boundary conditions are u(0) = 0and u(1) = 1, and the network architecture is 1–50–50–1. (a), (b) Training loss and normalized absolute error as a function of epochs for ˜ ubc nn (x; θ)and ˜ unn (x; θ). (c) Comparisons of ˜ ubc nn (x) and ˜ unn (x) with the exact solution. (d) Error in the displacement and strain fields. 

of the Gaussians and the weights in the output layer are the unknown parameters in the network. The centers for the neurons are chosen as b = {0, 1/9, 2/9, . . . , 1}. In Fig. 23, the numerical results for ˜ unn (x; θ) using collocation and Ritz are presented. By the end of the training, the loss for the collocation and Ritz solutions are O(10 −1) or better. Note that the loss measures are distinct for the collocation and Ritz solutions; PDE loss is shown for the former whereas it is the (Ritz) energy loss that is presented for the latter. The collocation solution is in good agreement with the exact solution (Fig. 23b). From Fig. 23c, we observe that the errors in the displacement field for both methods are small, but the errors in the derivative (strain fields) are appreciable. With more number of neurons in the hidden layer, it is expected that the accuracy in the strain field will substantially improve. This example reveals the flexibility that the PINN a ff ords in that variational adaptive solutions can be captured by a Gaussian neural network with a single hidden layer. Realizing this is much more di ffi cult using meshfree basis functions, since the underlying Ritz formulation becomes a nonlinear, nonconvex minimization problem. 0 2000 4000 Epochs 10 −1

> Training loss

Collocation Ritz 0.00 0.25 0.50 0.75 1.00 

x

0.25 0.50 0.75 1.00 1.25 1.50 

> u(x), ˜ubc nn (x)

exact ˜ubc nn (Coll.) 0.0 0.5 1.0

x

−1.0

−0.50.00.51.0Prediction errors 

u- ˜ uh (Coll.) 

u′- ˜ uh′ (Coll.) 

u- ˜ uh (Ritz) 

u′- ˜ uh′ (Ritz) 

(a) (b) (c) 

Figure 23: Collocation and Ritz solutions using a Gaussian activation function for a Dirichlet problem with the exact solution as the sum of two Gaussians. Network architecture is 1–10–1. (a) Training loss as a function of epochs for ˜ ubc nn (x; θ) (collocation and Ritz methods). (b) Comparisons of ˜ ubc nn (x) (collocation) with the exact solution. (c) Error in the displacement and strain fields for collocation and Ritz methods. 

9.2. Longitudinal vibrations of a homogeneous elastic rod 

The eigenproblem for the longitudinal vibrations of an elastic bar that is fixed at both ends is: 

u′′ + ω2u = 0 in Ω = (0 , 1) , (58a) 

u(0) = u(1) = 0. (58b) The exact eigenfunctions are: un(x) = sin( ωn x), where ωn = nπ (n ∈ N) are the natural frequencies. The eigenvalue 

λn = ω2 

> n

corresponds to the eigenfunction un(x). 27 We use the Ritz method to solve this problem using ˜ ubc nn (x; θ) and ˜ unn (x; θ). The Rayleigh quotient minimization problem for the smallest eigenvalue (lowest mode) is [9]: min 

> u∈S

∫ 10 u′2 dx 

∫ 10 u2 dx 

, subject to 

∫ 10

u2 dx = 1, (59) where S = {u : u ∈ H1(Ω), u(0) = u(1) = 0} and the normalization constraint on the eigenfunction appears in (59). For the trial function ˜ ubc nn (x; θ), the loss function is: 

Lbc nn (θ) =

∑NI

> k=1

(˜ubc nn ′(xk; θ))2

∑NI

> k=1

(˜ubc nn (xk; θ))2 +



1

NINI∑

> k=1

(˜ubc nn (xk; θ))2

 − 1



> 2

, (60) where NI is the number of interior integration points. Note that apart from the PDE loss term, we have an additional loss term due to the normalization constraint in (59). The loss function for ˜ unn (x; θ) consists of the two contributions that appears in (60), and in addition it will include two boundary loss terms to impose the essential boundary conditions. The network architecture 1–50–50–50–1 is used. In Fig. 24, the Ritz solutions for ˜ ubc nn (x) and ˜ unn (x) are presented. The loss function for ˜ ubc nn and ˜ unn saturate to values of 1 and 10, respectively. The error in the natural frequency for ˜ubc nn (x; θ) and ˜ unn (x; θ) are O(10 −4) and O(1) at 10,000 epochs (see Fig. 24b). In Fig. 24c, we compare the lowest mode (eigenfunction) from the numerical solutions to the exact solution: ˜ ubc nn (x) is in much better agreement with the exact solution than ˜ unn (x). The exact mode shape is well-captured by ˜ ubc nn (x) but it is has not been exactly normalized, which leads to the observed discrepancy in the maximum amplitude. 0 5000 10000 Epochs 246810 Training loss ˜ubc nn 

˜unn 

0 5000 10000 Epochs 10 −4

10 −3

10 −2

10 −1

10 0

> Error in frequency

˜ubc nn 

˜unn 

0.0 0.5 1.0

x

0.00.51.01.5Modeshape 

exact ˜ubc nn 

˜unn 

(a) (b) (c)      

> Figure 24: Ritz solutions for the longitudinal vibrations (lowest mode) of a homogeneous elastic rod. The network architecture is 1–50–50–50–1. (a), (b) Training loss and error in lowest natural frequency as a function of epochs for ˜ ubc nn (x;θ) and ˜ unn (x;θ). (c) Comparisons of ˜ ubc nn (x) and ˜ unn (x)for the lowest mode (eigenfunction) that corresponds to the lowest natural frequency ω1=π.

9.3. Advection-di ff usion problem u′′ = αu′ in Ω = (0 , 1) (61a) 

u(0) = 0, u(1) = 1, (61b) where α is the Peclet number, which measures the ratio of the advective rate to the di ff usion rate. The exact solution of the problem posed in (61) is: 

u(x) = eαx − 1

eα − 1 . (62) We choose α = 0, 5, 10 , 50 in this study (pure di ff usion for α = 0 to strongly advective flow for α = 50) and run collocation simulations using ˜ ubc nn (x; θ). For α = 1, 5, 10, the network architecture 1–50–50–1 is used, and for α = 50, the architecture is 1–50–50–50–1. In Fig. 25, the simulation results are presented. For all cases shown in Fig. 25b, we observe that the normalized absolute error during the training is O(10 −3) or less. For all α that are selected, Fig. 25c shows an excellent match between ˜ ubc nn (x) and the exact solutions. 28 0 10000 20000 Epochs 10 −12                     

> 10 −8
> 10 −4
> 10 0
> 10 4
> Training loss
> α=0
> α=5
> α=10
> α=50 010000 20000 Epochs 10 −9
> 10 −7
> 10 −5
> 10 −3
> 10 −1
> Error
> α=0
> α=5
> α=10
> α=50 0.00.51.0
> x
> 0.00.20.40.60.81.0
> u(x), ˜ubc nn (x)

(a) (b) (c)           

> Figure 25: Collocation solutions using ˜ ubc nn (x;θ) for the advection-di ff usion problem. For α=1,5,10, the network architecture 1–50–50–1 is used, and for α=50, the architecture is 1–50–50–50–1. (a), (b) Training loss and normalized absolute error as a function of epochs. (c) Comparisons of ˜ubc nn (x) to the exact solutions for di ff erent α. The solid lines are for the exact solutions and the markers (colors are consistent with those shown in (b)) represent the numerical solutions.

9.4. Euler-Bernoulli beam bending 

Consider the boundary-value problem for the deflection of a cantilever (Euler-Bernoulli) beam of unit length that is clamped at both ends and is subjected to a distributed load q(x): 

EIv ′′′′ = q in Ω = (0 , 1) , (63a) 

v(0) = v′(0) = v(1) = v′(1) = 0, (63b) where EI is the flexural rigidity of the beam. We apply a point moment (clockwise orientation) of magnitude M0 at x = 1/2 so that q(x) = M0δ′(x − 1/2). For this q(x), the variational principle that is associated with the strong form in (63) is: min 

> v∈S

[

Π[v] = 12

∫ 10

EI (v′′ )2dx + M0v′(1 /2) , S =

{

v : v ∈ H1(0 , 1) , v(0) = v′(0) = v(1) = v′(1) = 0

}] 

. (64) For this problem, all homogeneous boundary conditions associated with v and v′ that appear in (63b) are essential boundary conditions, and hence a kinematically admissible PINN trial function is given by (50): ˜vbc nn (x; θ) = [φ(x)] 2 ˜vR

> nn

(x; θ), (65) where φ(x) is an ADF (normalized to order 1) that vanishes at x = 0 and x = 1, and its normal derivative has unit magnitude on the boundary. For the computations, we choose EI = 1 and M0 = 1. The exact solution of (63) is: 

v(x) =

[ReLU (x − 1/2) ]2

2 + x2

8 − x3

4 , x ∈ Ω, (66) which is a C1(Ω) (piecewise cubic) function with a moment discontinuity of unit magnitude at x = 1/2. Deep Ritz solutions ˜ ubc nn (x; θ) and ˜ unn (x; θ) are computed on the network architecture 1–50–50–1 using the cubic ReLU activation function, and the results are presented in Fig. 26. From Fig. 26a, we observe that ˜ ubc nn (x; θ) converges to a loss of O(10 −4) at 20,000 epochs but the loss for ˜ unn (x; θ) remains at about 0.1. Figure 26b shows that the numerical solution ˜ ubc nn (x) is in excellent agreement with the exact solution. In Fig. 26c, the error fields are shown: the deflection and rotation fields using ˜ ubc nn (x) are accurate, whereas both fields have large errors for ˜ ubc nn (x). We attribute the poor performance of ˜ unn (x; θ) due to scaling issues of the interior and boundary terms in the loss function. If the ratio of the weights assigned to the boundary loss term and the interior loss terms is set to 10 3 : 1, we find that the results improve but they are still worse than those obtained using ˜ ubc nn (x). It requires a weight ratio of 10 4 for the two solutions to have comparable accuracy. 29 0 10000 20000 Epochs 10 −4      

> 10 −3
> 10 −2
> 10 −1
> Error  ˜ubc nn
> ˜unn
> 0.00 0.25 0.50 0.75 1.00
> x
> −0.004
> −0.002 0.000 0.002 0.004
> u(x), ˜ubc nn (x)
> exact ˜ubc nn
> 0.00.51.0
> x
> −1.0
> −0.50.00.5Prediction errors
> u- ˜ ubc nn
> u′- ˜ ubc nn ′
> u- ˜ unn
> u′- ˜ unn ′

(a) (b) (c)   

> Figure 26: Deep Ritz solutions using ˜ ubc nn (x;θ) and ˜ unn (x;θ) for the Euler-Bernoulli beam problem. The network architecture 1–50–50–1 is used. (a) Normalized absolute eror as a function of epochs. (b) Comparison of ˜ ubc nn (x) with the exact solution. (c) Errors in the deflection and rotation fields.

10. Numerical Examples in Two Dimensions 

The promise of PINN is intriguing for inverse and parameteric (design) problems. But this rests on its accuracy, robustness and reliability on solving the forward problem, which is the emphasis in this contribution. To this end, we focus on the performance of the PINN formulation with exact imposition of boundary conditions versus the standard PINN [6] with equally-weighted loss terms. For the two-dimensional problems, we consider polygonal domains and also domains with curved boundaries. We consider four distinct types of problems: steady-state heat conduction; computation of harmonic (Laplace equation) coordinates [99], which is an instance of generalized barycentric coor-dinates [80]; clamped Kirchho ff plate bending (fourth-order PDE); and the Eikonal equation to compute the signed distance function to a boundary. For these problems, we present our solutions and compare them to either the exact solution (if available) or a reference finite element solution, and to deep collocation [6]. In addition, we identify key di ff erential properties of the approximate distance function and bring to fore the issue of exact satisfaction of boundary conditions and its implications in the training of the network and the accuracy that the PINN approximation delivers. Prior to presenting the numerical examples, it is instructive to understand the properties and behavior of approxi-mate distance functions that are obtained by either R-functions with R-equivalence as presented in Section 3.2 or via mean value potential fields that are discussed in Section 4. Since these functions are used in the PINN ansatz ˜ ubc nn (x; θ)that is presented in Section 5, one must consider the regularity of these functions when used in a deep collocation or a deep Ritz method. 

10.1. Laplacian of approximate distance fields 

For Poisson or Laplace boundary-value problems that involve the Laplace operator, we must understand the be-havior of the Laplacian of the ADFs that stem from R-functions and mean value potentials. Let us consider the unit square, Ω = (0 , 1) 2. The boundary ∂Ω consists of four line segments. On using either (10) or (14), we can construct an approximate distance function to the boundary, φ(x), which is normalized to order 1. Let us refer to these functions as 

φR(x) (REQ) and φM (x) (MVP). In Fig. 27, φ(x) and its Laplacian over the unit square are presented. We observe that both φR and φM are zero on the entire boundary and monotonic (concave) inside the domain. This property of these functions on ∂Ω is used to impose essential boundary conditions, as described in Section 5.2. From Figs. 27c and 27d, we observe that the Laplacians, ∇2φR and ∇2φM , blow up at the vertices of the square. In fact, it is known that both 

∇2φR and ∇2φM are singular at the vertices of a polygon, and therefore very large in magnitude near any of its vertices. Therefore, in a collocation-based approach to solve the Poisson equation, which involves the Laplacian, the contribu-tions to the total loss from regions near the vertices can be very large. This inference does not influence Ritz-based solutions of the Poisson equation since the highest derivative in the variational principle is of order 1, and both φR and 

φM and its first-order derivatives are well-behaved (bounded) over the entire domain. There are two possible remedies to address this issue. The first involves modifying the φi that are obtained from R-equivalence and the ADFs that stem from mean value potential fields, so that the corresponding Laplacians are bounded in the domain. The second is to consider collocation points inside the domain that are not very close to the vertices. We leave the first route as part of future work, and proceed in this paper with the second choice. For instance, we show in Section 10.2.2 that if all 30 interior collocation points are located in Ωδ = [δ, 1 − δ]2 (δ = 0.01), which is a subset of the unit square, then both methods perform well. Finally, we reemphasize that it is imperative that in most instances φ be smooth in the interior of a computational domain; otherwise, ∇2φ will blow up at an interior collocation point and then one cannot use a trial function that uses φ in a collocation-based PINN method. So in most instances in 2D or 3D, this precludes the use of exact distance functions in the ansatz, and hence approximate distance functions should be used. To show this, let us consider the exact signed distance to the unit disk, φ(x) = 1 − √x2 + y2. We can write 

∂2φ∂x2 = x2

(x2 + y2)3/2 − 1

√x2 + y2 ,

and since the second term is unbounded at the origin, the Laplacian of φ blows up at the origin. There are exceptions when the exact distance function can be used. It is a suitable choice when the medial axis of a domain is not part of the computational domain. For example, when solving a boundary-value problem over an annulus in 2D (see the problem solved in Section 10.2.3) or a hollow cylinder in 3D, then the exact distance function can be used since the origin (where the exact distance function has derivative discontinuities) lies outside the computational domain. 0.0 0.51.0 0 .00.51.00.05 0.10 0.0 0.51.0 0 .00.51.00.025 0.050 0.075 0.0 0.51.0 0 .00.51.0 

> −40
> −20 0.00.51.0 0 .00.51.0
> −30
> −20
> −10

(a) φR(x) (b) φM (x) (c) ∇2φR(x) (d) ∇2φM (x)  

> Figure 27: Computation of φ(x) and ∇2φ(x) over the unit square for ADFs constructed from R-equivalence and mean value potential fields.

10.2. Steady-state heat conduction 

Let us consider the following model problem for isotropic steady-state heat conduction: 

−∇ 2u = f in Ω ⊂ R2 (67a) 

u = g on Γu, ∂u

∂n + cu = h on Γn (67b) where u(x) is the temperature field and f (x) is the heat source. The boundary ∂Ω = Γ u ∪ Γn is partitioned into two parts, with Γn ∩ Γn = ∅. The temperature field g(x) is imposed on the essential boundary Γu, and boundary data h(x)that is associated with a Robin boundary condition is prescribed on Γn (c is in general a spatially varying field). 

10.2.1. Essential boundary conditions 

As the first example, we consider the biunit square, Ω = (−1, 1) 2, with u = g = 0 prescribed on ∂Ω. If k ∈ N and 

f (x, y) = sin( kπx) sin( kπy) is the forcing function, then the exact solution for this problem is: 

u(x) = sin( kπx) sin( kπy)2k2π2 .

In the numerical computations, we consider two distinct forms of trial functions in the neural network. The first ansatz is the standard PINN that is given by ˜ unn (x; θ), which does not a priori satisfy the boundary condition. The second form consists of trial functions that are given by ˜ ubc nn (x; θ) = φ(x)˜ uR

> nn

(x; θ), where φ(x) is a function that is zero on ∂Ω.This property of φ(x) ensures that ˜ ubc nn (x; θ) automatically satisfies the essential boundary conditions. While an obvious choice for φ(x) is (1 − x2)(1 − y2), here we consider φ(x) that are constructed using R-functions with R-equivalence composition (see Section 3.2) and by mean value potential fields (see Section 4) as they readily generalize to more 31 complex domains. When needed, we use the acronyms REQ and MVP to distinguish the numerical solutions, ˜ ubc nn (x), which are obtained using these two methods. The plot of φ(x) using REQ and MVP over a square are shown in Figures 7b and 10a, and it can be observed that φ is zero on the boundary of the domain. For the collocation scheme, we randomly sample NI number of points in Ω and NB number of points on ∂Ω. To solve the problem using standard PINN, we minimize the loss Lnn (θ) given in (41), which is reproduced below (using g = 0): 

Lnn (θ) = ||∇ 2 ˜unn (x; θ) + f (x)|| 2 

> Ω,NI

+ || ˜unn (x; θ)|| 2 

> Ω,NB

,

where || · || Ω,NI and || · || ∂Ω,NB are defined in (40) and (41). Since ˜ ubc nn (x; θ) automatically satisfies the boundary condition, the parameters in this ansatz are found by minimizing the loss given in (40), which for this problem is: 

Lbc nn (θ) = ||∇ 2 ˜ubc nn (x; θ) + f (x)|| 2 

> Ω,NI

.

Figure 28 shows the training loss and normalized absolute error as functions of the training epochs for k = 1, 2(NI = 5, 000 , NB = 400). We observe from Figs. 28a and 28c that the training loss for ˜ unn (x; θ) is either comparable to or less than the loss for ˜ ubc nn (x; θ) over the same number of epochs. However, this does not translate into better prediction accuracy. Figures 28b and 28d show the normalized absolute errors for ˜ ubc nn (x; θ) (REQ and MVP) as well as ˜ unn (x; θ). These plots reveal that both REQ- and MVP-based schemes deliver an order of magnitude more accurate solutions compared to ˜ unn (x; θ). 0 5000 10000 Epochs 10 −4      

> 10 −3
> 10 −2
> 10 −1
> Training loss REQ MVP ˜unn
> 05000 10000 Epochs 10 −3
> 10 −2
> 10 −1
> 10 0
> Error REQ MVP ˜unn
> 05000 10000 Epochs 10 −3
> 10 −2
> 10 −1
> Training loss REQ MVP ˜unn
> 05000 10000 Epochs 10 −1
> 10 0
> Error  REQ MVC ˜unn

(a) k = 1 (b) k = 1 (c) k = 2 (d) k = 2         

> Figure 28: Training loss and normalized absolute errors for ˜ ubc nn (x;θ) (REQ and MVP) and ˜ unn (x;θ) in the heat conduction problem with homoge-neous Dirichlet boundary conditions and forcing function sin kπx. (a), (b) k=1, and (c), (d) k=2.

It is interesting to note that ˜ ubc nn (x; θ) produces smaller normalized absolute errors during training than ˜ unn (x; θ)even though it has larger losses. This observation is noticed in almost all cases, and it deserves some comments here. We mention that there is no reason to assume that the losses for ˜ ubc nn (x; θ) and ˜ unn (x; θ) should be comparable. For the problem under consideration, Lbc nn (θ) comprises of terms that involve the derivatives of φ(x) and these terms are not present in Lnn (θ). A further issue with ˜ unn (x; θ) is the relative scaling of the losses in Lnn (θ), which comprises of a loss on ∇2 ˜unn (x; θ) and a loss on ˜ unn (x; θ). In this problem, the norm of u is much smaller than the norm of its Laplacian, and therefore the ∇2 ˜unn (x; θ) term dominates in Lnn (θ). However, the optimizer can drive Lnn (θ) to very small values without adequately addressing the boundary loss term. We can see that this is indeed the issue if we compare the prediction errors for ˜ unn (x) over the domain (see Fig. 29). We see that the prediction errors for REQ and MVP schemes are much smaller than ˜ unn (x), and furthermore, the prediction errors for ˜ unn (x) are large near the boundary of the domain. This suggests that ˜ unn (x; θ) is undervaluing the boundary loss term in Lnn (θ). In principle, it is possible to improve the results for ˜ unn (x; θ) by assigning a larger weight to the boundary loss term in Lnn (θ) [16], but this is an ad-hoc remedy, which is not needed in our approach since the boundary condition is exactly met. To see that this is indeed the case, one can assume the loss function for ˜ unn (x; θ) to be a convex combination of the interior and boundary loss terms: 

Lnn (θ) = w ||∇ 2 ˜unn (x; θ) + f (x)|| 2

> Ω,NI

︸ ︷︷ ︸

> PDE loss

+ (1 − w) || ˜unn (x; θ)|| 2

> ∂Ω,NB

︸ ︷︷ ︸

> Boundary loss

,

where w ∈ [0 , 1] is a scalar that can be used to tune the relative weights of the two losses. Figures 30a–30c show the evolution of the PDE loss and the boundary loss for di ff erent values of w (k = 1). The value w = 0.1 weighs the boundary loss term 9 times more than the PDE loss term and it achieves smaller 32 −1 0 1 −101

> ×10 −4

01

−1 0 1 −101

> ×10 −4

−2

−101

−1 0 1 −101

> ×10 −3

−101(a) (b) (c) 

Figure 29: Prediction errors for k = 1 in Example 1. (a) ˜ ubc nn (x) (REQ), (b) ˜ ubc nn (x) (MVP), and (c) ˜ unn (x). 

error than both w = 0.5, 0.9 (Fig. 30d). This suggests that there likely is a sweet spot for w that results in very low errors. However, the exact regime for this solution is likely dependent on the problem under consideration and on the boundary conditions, the determination of which may be impossible to ascertain in problems where the exact solution is unknown. A distinguishing attribute of our approach is that no such tuning of relative weights is needed. 0 5000 10000 Epochs 10 −18             

> 10 −14
> 10 −10
> 10 −6
> 10 −2
> PDE loss Boundary loss 05000 10000 Epochs 10 −20
> 10 −15
> 10 −10
> 10 −5
> 10 0
> PDE loss Boundary loss 05000 10000 Epochs 10 −8
> 10 −6
> 10 −4
> 10 −2
> PDE loss Boundary loss 05000 10000 Epochs 10 −2
> 10 −1
> 10 0
> w=0.5
> w=0.1
> w=0.9

(a) w = 0.5 (b) w = 0.1 (c) w = 0.9 (d) Error 

Figure 30: Loss function for ˜ unn (x; θ) in Example 1 is a convex combination of PDE loss and boundary loss terms ( k = 1). Evolution of loss function for (a) w = 0.5, (b) w = 0.1 and (c) w = 0.9. (d) Evolution of normalized absolute error during training for w = 0.5, 0.1, 0.9. 

10.2.2. Example 2 

We consider the Laplace equation ( f = 0) over the unit square, Ω = (0 , 1) 2, with boundary conditions 

u(x) = 0 on Γ1, Γ2, Γ3, u(x) = g4(x) = sin πx on Γ4, (68) where Γ1 = {(x, y) : x = 0, 0 ≤ y ≤ 1}, Γ2 = {(x, y) : 0 ≤ x ≤ 1, y = 0}, Γ3 = {(x, y) : x = 1, 0 ≤ y ≤ 1}, and 

Γ4 = {(x, y) : 0 ≤ x ≤ 1, y = 1} are the boundary edges. The exact solution for this problem is: 

u(x) = (e−πy + eπy) sin πxe−π + eπ .

We chose this problem to demonstrate how to exactly satisfy nonzero essential boundary conditions on di ff erent subsets of the boundary through the use of transfinite interpolation. To construct a trial solution that satisfies the boundary conditions, we first create a composite approximate distance function, φ(x), to Γ = Γ 1 ∪ Γ2 ∪ Γ3 ∪ Γ4.This ADF can either be formed by the joining operation via R-equivalence, or directly via (14) that uses mean value potential fields on polygons. The resultant φ(x) is similar to the φ(x) used in Example 1 (see Fig. 27). We combine the Dirichlet boundary data into one function g(x) by using the transfinite interpolant in (26). In Fig. 31, the function g(x) and its Laplacian are plotted over the unit square. We observe that g(x) is zero on Γα

(α = 1, 2, 3), and it is equal to sin πx on Γ4. Referring to (27), the trial function for PINN is: ˜ubc nn (x; θ) = g(x) + φ(x) ˜ uR

> nn

(x; θ).

33 Since g(x) satisfies the boundary conditions in (68) and φ = 0 on Γ, ˜ ubc nn (x; θ) satisfies the Dirichlet boundary conditions on all edges of the boundary. From Fig. 31b, we observe that the Laplacian of g(x) is singular at two of the four vertices on the boundary and again this is handled by performing collocation over the smaller square [0 .01 , 0.99] 2.x

0.0 0.2 0.4 0.6 0.8 1.0

y

0.00.20.40.60.81.00.20.40.60.8

x

0.0 0.2 0.4 0.6 0.8 1.0

y

0.00.20.40.60.81.0050 100 150 

(a) g(x) (b) ∇2g(x)  

> Figure 31: Plots of (a) g(x) and (b) ∇2g(x) over the unit square.

We determine the parameters of this network by minimizing Lbc nn (θ) as described earlier (with f set to zero). For standard PINN, the parameters of ˜ unn (x; θ) are determined by minimizing the following loss function: 

Lnn (θ) = ||∇ 2 ˜unn (x; θ)|| 2 

> Ω,NI

+

> 4

∑

> α=1

|| ˜unn (x; θ) − gα(x)|| 2

> Γα,Nα
> B

,

where Nα 

> B

is the number of collocation points on Γα (α = 1, 2, 3, 4) and gα(x) = 0 ( α = 1, 2, 3, 4) in this problem. In the computations, we select a total of 400 boundary collocation points and 5,000 interior collocation within [ .01 , . 99] 2.The numerical results are shown in Fig. 32, where the training loss, the normalized absolute error, and exact and approximate solutions are presented for ˜ ubc nn (x; θ) and ˜ unn (x; θ). As in the previous example, here also we notice that the training loss for ˜ unn (x; θ) is orders of magnitude smaller than both REQ and MVP (Fig. 32a). The losses for ˜ubc nn (x; θ) start at relatively high values during the initial stages of the training, which is due to the large contributions from the Laplacian in the vicinity of the vertices. However, the network is quickly able to optimize and bring the losses down by almost two orders of magnitude in a couple of thousand training epochs. Still, the losses for ˜ ubc nn (x; θ)remain several orders of magnitude larger than ˜ unn (x; θ) at the end of the training. However, this example also reveals that the absolute value of the loss in itself is not very meaningful. The normalized absolute errors of the three schemes are presented in Fig. 32b as a function of the training epochs. It is evident from this plot that the errors in ˜ ubc nn (x; θ) are orders of magnitude smaller than the error in ˜ unn (x; θ). The error achieved by ˜ ubc nn (x; θ) (REQ) is almost an order of magnitude smaller than ˜ ubc nn (x; θ) (MVP), and almost two orders of magnitude better than ˜ unn (x; θ). Contour plots of the exact and approximate solutions appear in Figs. 32c–32f. The predicted errors of the three schemes are displayed in Fig. 33. It can be seen that the boundary errors in both REQ and MVP are precisely zero. This is expected, since ˜ ubc nn (x; θ) has been designed to satisfy the boundary conditions. On the other hand, ˜ unn (x; θ) has large errors on the boundary of the domain. The errors from ˜ unn (x; θ)are roughly an order of magnitude larger than MVP and two orders of magnitude larger than REQ. As in the previous example, one can improve the results of ˜ unn (x; θ) by weighing the boundary loss more than the PDE loss in Lnn (θ). This problem can also be solved using a Ritz scheme, which is appealing since only first-order derivatives are required in the loss function. We form a trial function that satisfies the essential conditions using (27): ˜ubc nn (x; θ) = g(x) + φ(x) ˜ uR

> nn

(x; θ).

The parameters of the network can now be found by minimizing the loss in (47). To numerically evaluate the integral, we divide the square into a uniform grid with NI number of interior points. Since the Ritz loss does not involve second 34 0 5000 10000 Epochs 10 −2          

> 10 0
> 10 2
> Training loss REQ MVP ˜unn
> 05000 10000 Epochs 10 −3
> 10 −2
> 10 −1
> 10 0
> Error REQ MVP ˜unn
> 0.00.51.0
> x
> 0.00.20.40.60.8
> y
> 0.00.51.0
> x
> 0.00.20.40.60.8
> y
> 0.00.51.0
> x
> 0.00.20.40.60.8
> y
> 0.00.51.0
> x
> 0.00.20.40.60.8
> y
> 0.00 0.15 0.30 0.45 0.60 0.75 0.90

(a) (b) 

(c) (d) (e) (f) 

Figure 32: Numerical results for the Laplace problem on the unit square with nonzero essential boundary conditions. (a), (b) Training loss and normalized absolute errors for ˜ ubc nn (x; θ) (REQ and MVP) and ˜ unn (x; θ). Contour plots over the unit square of the (c) exact solution, (d) ˜ ubc nn (x)(REQ), (e) ˜ ubc nn (x) (MVP), and (f) ˜ unn (x). x

> 0.0 0.2 0.4 0.6 0.8 1.0
> y
> 0.00.20.40.60.81.0
> −0.0005 0.0000 0.0005
> x
> 0.0 0.2 0.4 0.6 0.8 1.0
> y
> 0.00.20.40.60.81.0
> −0.002 0.000 0.002 0.004
> x
> 0.0 0.2 0.4 0.6 0.8 1.0
> y
> 0.00.20.40.60.81.0
> −0.02 0.00 0.02

(a) (b) (c) 

Figure 33: Surface plots of the errors in the numerical solutions for the Laplace problem with nonzero essential boundary condition. (a) ˜ ubc nn (x)(REQ), (b) ˜ ubc nn (x) (MVP), and (c) ˜ unn (x). 

derivatives of φ(x) or g(x), all terms in the loss are well-defined and bounded even arbitrarily close to the boundaries of the domain. We select 5,000 interior points on the square [0 .0001 , 0.9999] 2. For a Dirichlet problem, it is especially important to sample close to the boundaries, because in the absence of doing so, the loss may be trivially minimized by a u(x) that is a constant. Sampling close to the boundaries informs the algorithm that a constant u(x) leads to large errors near the boundaries, which are manifested in the loss term as large gradients. Numerical results for the Ritz method using ˜ unn (x; θ) with REQ are presented in Fig. 34. The training loss and normalized absolute errors as a function of epochs are shown in Fig. 34a, which reveal that the error reduces to O(10 −2) in less than 2,000 epochs. In Fig. 34b, the prediction errors over the square are displayed. Compared to Fig. 33, we find that the errors in the Ritz scheme are smaller than MVP-based collocation but larger than REQ-based collocation. 35 0 2500 5000 7500 10000 Epochs 10 −2

10 −1

10 0

> Training loss and error

Loss Error 0.00 0.25 0.50 0.75 1.000 .00 0.25 0.50 0.75 1.00 

> ×10 −3

−5.0

−2.50.02.5(a) (b)  

> Figure 34: Ritz solution using ˜ ubc nn (x;θ) (REQ) for the Laplace problem in Example 2. (a) Evolution of training loss and normalized absolute error, and (b) Prediction errors over the domain after training.

10.2.3. Curved domain 

For a problem with a curved domain, we consider the Laplace equation, ∇2u = 0, in an annulus that is bounded between circles of radii, R1 = 1 (boundary Γ1) and R2 = 1/4 (boundary Γ2) [71]. The essential boundary conditions are: 

u = 1 on Γ1, u = 2 on Γ2.

The exact solution for this problem is: 

u(x) = 1 − ln √x2 + y2

ln 4 . (69) To impose the boundary conditions, we need a distance function to both Γ1 and Γ2, and also a composite boundary data function g(x). In this case, on using the exact distance functions to the two circles, we have φ1(x) = 1 − √x2 + y2

(positive in the interior of the larger disk), and φ2(x) = √x2 + y2 − 1/4 (positive outside the smaller disk). Since the origin is not part of the computational domain, here we can use the exact distance functions to form φ1 and φ2. Now, on combining these two ADFs using the R-equivalence operation ( m = 1) in (9), we obtain φ(x) = φ1 ∼ φ2 (positive in the annulus). Finally, we use the transfinite formula (26) to construct the composite boundary data as 

g(x) = 2φ1 + φ2

φ1 + φ2

= 7 − 4 √x2 + y2

3 .

In Fig. 35, φ(x) and g(x) are plotted over the annulus. To clearly see that φ(x) is zero on Γ1 and Γ2, we show −φ(x)in Fig. 35a. From the plot in Fig. 35b, we observe that g(x) matches the imposed boundary data on Γ1 and Γ2. An ansatz that exactly satisfies the boundary conditions is: ˜ ubc nn (x; θ) = g(x) + φ(x) ˜ uR

> nn

(x; θ). We compare the performance of ˜ ubc nn (x; θ) with a standard PINN trial function, ˜ unn (x; θ), where the boundary conditions need to be enforced through the loss function. The loss function for the two cases are similar to those discussed in previous examples, the only di ff erence being that now we sample the boundary data at NB points on the curved boundary Γ = Γ 1 ∪ Γ2 and the interior collocation data at NI points in Ω1 ∩ Ω2. We use dmsh [95] to triangulate the annulus and choose the centroid of the triangles as interior collocation points and the center of the edges on the boundary as the boundary collocation points. For this problem, we pick NI = 612 points in the interior of the domain, 66 points on Γ1 and 30 points on Γ2 for a total of NB = 96 points on Γ1 ∪ Γ2. A representative mesh that display the interior collocation points over the annulus is shown in Fig. 15. We use the Adam optimizer with a learning rate (step size) of 10 −3 for training both ˜ ubc nn (x; θ) and ˜unn (x; θ), and the training is stopped at 10,000 epochs for both networks in order to perform a fair comparison. In Fig. 36, the results of the training as well as the approximate solutions produced by ˜ ubc nn (x; θ) (REQ) and ˜ unn (x; θ)are presented. It is pertinent to mention here that ˜ unn (x; θ) required a much larger network architecture (2–150–150–1) compared to ˜ ubc nn (x; θ), which only required a 2–50–50–1 network in order to converge to acceptable results. However, even with a much larger network, as observable in Figs. 36a and 36b, the error in ˜ unn (x; θ) by the end of the training is 36 −1.0−0.50.0 0.5 1.0−1.0

−0.50.00.51.0

−0.15 

−0.10 

−0.05 0.00 

−1.0−0.50.0 0.5 1.0−1.0

−0.50.00.51.01.01.21.41.61.82.0(a) (b) 

Figure 35: Approximate distance function to the boundaries of the annulus using R-equivalence composition of the exact distance functions to Γ1

and Γ2. Plots of (a) −φ(x) and (b) g(x) that interpolate essential boundary data on Γ1 and Γ2.

two orders of magnitude larger than ˜ ubc nn (x; θ). It can be seen from Fig. 36e that ˜ unn (x; θ) does not satisfy the boundary conditions. The approximation errors of ˜ ubc nn (x) (REQ) and ˜ unn (x) appear in Figs. 36f and 36g. The errors in ˜ unn (x) are especially large on the boundaries of the domain. 0 2500 5000 7500 10000 Epochs 10 −3      

> 10 −2
> 10 −1
> 10 0
> Training loss REQ (2-50-50-1) ˜unn (2-150-150-1) 02500 5000 7500 10000 Epochs 10 −3
> 10 −2
> 10 −1
> 10 0
> Error  REQ (2-50-50-1) ˜unn (2-150-150-1)
> −1.0
> −0.50.0 0.5 1.0−1.0
> −0.50.00.51.01.01.21.41.61.82.0
> −1.0
> −0.50.0 0.5 1.0−1.0
> −0.50.00.51.01.01.21.41.61.82.0
> −1.0
> −0.50.0 0.5 1.0−1.0
> −0.50.00.51.01.21.41.61.8
> −1.0
> −0.50.0 0.5 1.0−1.0
> −0.50.00.51.0
> ×10 −4
> −7.5
> −5.0
> −2.50.02.5
> −1.0
> −0.50.0 0.5 1.0−1.0
> −0.50.00.51.0
> ×10 −1
> −1.0
> −0.50.00.5

(a) (b) (c) (d) (e) (f) (g) 

Figure 36: Numerical solutions for the Laplace problem on an annulus with Dirichlet boundary conditions. (a), (b) Training loss and normalized absolute errors of ˜ ubc nn (x; θ) (REQ) and ˜ unn (x; θ). Surface plots of the (c) exact solution, (d) ˜ ubc nn (x; θ) using 2–50–50–1 network, and (e) ˜ unn (x) using 2–150–150–1 network. Surface plots of the error for the numerical solutions (f) ˜ ubc nn (x) and (g) ˜ unn (x). 

We now use the Laplace problem over the annulus to also demonstrate how to solve the problem using mixed boundary conditions. To this end, we retain the essential boundary condition on Γ1 and convert the boundary condition on Γ2 to a Robin boundary condition. For this problem, we find that ∂u/∂ n = 4/ ln 4 on the inner boundary. We use 37 the following mixed boundary conditions: 

u = 1 on Γ1, ∂u

∂n + u = 2 + 4ln 4 =: h on Γ2. (70) The exact solution remains unchanged and is given in (69). To create an ansatz for this problem with mixed boundary conditions, we follow the formulation in Section 5.3. We form φ1, φ2, which remain unchanged from the previous case when essential boundary conditions are imposed on Γ1 and Γ2. Since g = 1, referring to (35), we can write 

u1(x) = g(x) = 1, u2(x) = [1 + φ2(1 + Dφ2 

> 1

)] (˜ uR

> nn

(x; θ)) − φ2h, (71) where h is given in (70), and then on using transfinite interpolation given in (35a) and (35b), we form the trial function ˜ubc nn (x; θ). So we now have an ansatz that satisfies both the Dirichlet and Robin boundary conditions. The training loss and normalized absolute errors for the numerical solution using ˜ ubc nn (x; θ) are presented in Figs 37a and 37b. On comparing ˜ ubc nn (x) in Fig. 37c to the exact solution shown in Fig. 36, we see that the boundary conditions have been exactly satisfied. The error plot in Fig. 37d reveals that the numerical solution is within 1 percent of the exact solution over the domain. 0 10000 20000 30000 Epochs 10 0   

> 10 1
> 10 2
> Training loss 010000 20000 30000 Epochs 10 −2
> 10 −1
> 10 0
> Error
> −1.0−0.50.0 0.5 1.0−1.0
> −0.50.00.51.01.01.21.41.61.82.0
> −1.0−0.50.0 0.5 1.0−1.0
> −0.50.00.51.0
> −0.010
> −0.005 0.000 0.005 0.010

(a) (b) (c) (d)  

> Figure 37: Numerical solution using ˜ ubc nn (x;θ) (REQ) for the Laplace problem in an annulus with mixed boundary conditions. (a), (b) Training loss and normalized absolute errors as a function of epochs. Surface plots of (c) ˜ ubc nn (x) and (d) error in ˜ ubc nn (x).

10.3. Generalized barycentric coordinates over polygons 

Consider a planar polygon with n vertices (nodal coordinate {xi}ni=1) that are in counterclockwise orientation. On an n-gon, harmonic coordinates [99] are one of the instances of generalized barycentric coordinates [79]. Each coordinate (shape function), ϕi := ϕi(x), is associated with vertex i and is obtained by solving the Laplace equation with piecewise a ffi ne Dirichlet boundary conditions. The boundary-value problem for harmonic coordinates is: find 

φi ≥ 0 ( i = 1, 2, . . . , n) that solves 

∇2ϕi = 0 in Ω, (72a) 

ϕi = gi on ∂Ω, (72b) where gi := gi(x) is a piecewise a ffi ne (hat) function that is unity at xi and is zero at all other vertices, i.e., gi(x j) = δi j ,where δi j is the Kronecker-delta. By virtue of the maximum principle for the Laplace equations, φi > 0 in the interior of the polygon. Here we will solve the harmonic coordinate problem on two representative polygons: a square and an L-shaped (nonconvex) polygon. For both examples, the network architecture 2–50–50–1 is used. In both examples, an im-portant step is to assemble the boundary data into a function g(x) through transfinite interpolation. Figure 38 shows the function g(x) for specific choices of the vertex i. If the vertices of the square are numbered 1–2–3–4 (coun-terclockwise), starting at vertex 1 that is at (0 , 0), then the harmonic coordinate u(x) must satisfy u(0 , 0) = 1, 

u(1 , 0) = u(1 , 1) = u(0 , 1) = 0. So the boundary conditions are a ffi ne along edges 1–2 and 4–1 and identically zero along edges 2–3 and 3–4. All these boundary conditions are simultaneously captured in the g(x) function. It can be seen from the colormaps that g(x) appropriately interpolates the boundary data in all cases. 38 0.2 0.4 0.6 0.80.20.40.60.8       

> −0.50.00.5
> −0.8
> −0.6
> −0.4
> −0.20.00.20.40.60.8
> 0.00.20.40.60.81.00.00.20.40.60.81.0
> 0.00 0.15 0.30 0.45 0.60 0.75 0.90

(a) (b) (c)  

> Figure 38: Computation of harmonic coordinate for a vertex in a polygon. Contour plots of g(x) over a (a) square (vertex at the origin), (b) regular hexagon (rightmost vertex), and (c) L-shaped polygon (vertex at the origin).

10.3.1. Harmonic coordinates on a square 

We first consider the case of computing the harmonic coordinate over a square. On a square, there exists an exact solution for this problem—harmonic coordinates coincide with bilinear finite element shape functions. So the solution that is associated with vertex 1 is: u(x, y) = (1 − x)(1 − y). To compute the harmonic coordinates, we adopt the Ritz method to determine the approximate solutions. Since harmonic coordinates minimize the Dirichlet energy, use of the Ritz formulation is natural. As done earlier, we consider the approximations ˜ ubc nn (x; θ) and ˜ unn (x; θ). For ˜ubc nn (x; θ), the loss to be minimized is given in (47), whereas for ˜ unn (x; θ), it is supplemented with an additional term 

∑n 

> α=1

|| ˜unn (x; θ)−gα(x)|| 2

> ∂Ωα,Nα
> B

to impose the essential boundary conditions. Here, n is the number of boundary segments over which the Dirichlet boundary conditions are specified. In Fig. 39, numerical results using ˜ ubc nn (x; θ) (REQ) and ˜ unn (x; θ) are presented. In Figs. 39a and 39b, the training loss and normalized absolute error as a function of epochs are presented. The exact solution along with the numerical solutions are displayed in Figs. 39c–39e. Surface plots of the errors in ˜ ubc nn (x) and ˜ unn (x) appear in Figs. 39f and 39g. Once again, and consistent with prior findings, ˜ ubc nn (x; θ) exactly satisfies the essential boundary conditions and has far smaller errors than ˜ unn (x; θ). The large errors in ˜ unn (x; θ) are particularly noticeable in Fig. 39g. The issue again has to do with the relative scaling of the PDE loss versus the boundary loss. As done earlier, the numerical results from ˜unn (x; θ) can be improved by considering a loss of the form: 

Lnn (θ) = w



1

NINI∑

> k=1

[ 12 a (˜unn (xk; θ), ˜unn (xk; θ)) − ` (˜unn (xk; θ))]  + (1 − w)



> n

∑

> α=1

|| ˜unn (x; θ) − gα(x)|| 2

> ∂Ωα,Nα
> B

 ,

and then tuning the weight w ∈ [0 , 1]. This tunes the relative importance of the boundary loss term with respect to the PDE loss term. In fact, if one considers w = 10 −3, ˜ unn (x; θ) achieves errors that are comparable to ˜ ubc nn (x; θ). In other words, the boundary loss in Lnn (θ) has to be weighed a thousand times more than the PDE loss in order for ˜ unn (x; θ)to produce results comparable to ˜ ubc nn (x; θ). 

10.3.2. Harmonic coordinates on an L-shaped polygon 

We repeat the computations for the square over an L-shaped polygon. Here we consider φ(x) that is formed using REQ and MVP. The plot of φ(x) for REQ and MVP over the L-shaped polygon are shown in Figs. 7d and 10b. The function g(x) over the L-shaped polygon is shown in Fig. 38c. We only present numerical simulation results for REQ and MVP. As shown for the case of the square, the loss terms have to be weighed judiciously to obtain acceptable accuracy for ˜ unn (x). Harmonic coordinates associated with the vertex at the origin and for the vertex at the reentrant corner are computed. Since an exact analytical solution for this problem is not available, we compute an accurate finite element solution that we use as the reference solution. This finite element solution is used to compute the errors in ˜ubc nn (x). A Delaunay triangular mesh is created using the mesh generation package Triangle [100]: mesh has 13,952 elements with very small elements in the vicinity of the reentrant corner and larger elements near other vertices. The mesh size, h = 10 −3, is used near the reentrant corner to capture the weakly singular behavior of the Laplace equation at the reentrant corner. 39 0 2500 5000 7500 10000 Epochs 10 −1            

> 2×10 −1
> 3×10 −1
> 4×10 −1
> 6×10 −1
> Training loss REQ ˜unn
> 02500 5000 7500 10000 Epochs 10 −2
> 10 −1
> 10 0
> Error  REQ ˜unn
> 0.00 0.25 0.50 0.75 1.000 .00 0.25 0.50 0.75 1.00 0.20.40.60.80.00 0.25 0.50 0.75 1.000 .00 0.25 0.50 0.75 1.00 0.00.20.40.60.80.00 0.25 0.50 0.75 1.000 .00 0.25 0.50 0.75 1.00 0.20 0.25 0.30 0.35 0.00 0.25 0.50 0.75 1.000 .00 0.25 0.50 0.75 1.00 0.000 0.002 0.004 0.00 0.25 0.50 0.75 1.000 .00 0.25 0.50 0.75 1.00
> −0.6
> −0.4
> −0.20.00.2

(a) (b) (c) 

(d) (e) (f) (g) 

Figure 39: Computation of harmonic coordinates on the unit square for the vertex at the origin. (a), (b) Training loss and errors for ˜ ubc nn (x; θ) (REQ) and ˜ unn (x; θ). Surface plots of the (c) exact solution, (d) ˜ ubc nn (x), and (e) ˜ unn (x). Surface plot of the error for the numerical solutions (f) ˜ ubc nn (x) and (g) ˜ unn (x). 

A representative mesh that displays the interior collocation points in the L-shaped polygon is shown in Fig. 15. In Figs. 40 and 41, numerical solutions obtained from ˜ ubc nn (x; θ) (REQ and MVP) are presented for the computation of 

u that is associated with vertices at (0 , 0) and (1 /2, 1/2), respectively. In Fig. 40a, we observe that the training loss stabilizes within a few thousand epochs to about 1.5 for REQ and MVP, and this correspond to a normalized absolute error (see Fig. 40b) of O(10 −2). The reference finite element solution is presented in Fig. 40c, and the error in ˜ ubc nn (x)(REQ and MVP) are displayed in Figs. 40d and 40e. Numerical solutions using REQ and MVP have maximum errors of about 3 percent. From Fig. 41a, we observe that the training loss stabilizes within a few thousand epochs to about 8 for REQ and MVP, and this correspond to a normalized absolute error (see Fig. 41b) of O(10 −1). The reference finite element solution is depicted in Fig. 41c, which display sharp gradients near the rentrant corner. The error in ˜ ubc nn (x)(REQ and MVP) are shown in Figs. 41d and 41e, with maximum errors near the singularity on the order of 20 percent. Compared to the errors in Figs. 40d and 40e, this is a 10-fold increase in the maximum error (due to the presence of the derivative singularity at the reentrant corner). 

10.4. Clamped circular Kirchho ff plate 

We consider the boundary-value problem for a clamped plate that is given in (48). For a clamped circular plate of unit radius and transverse load f = 1, the boundary-value problem is: 

∇4u = 1 in Ω = {(x, y) : x2 + y2 < 1}, (73a) 

u = 0 on ∂Ω, u, n := ∂u

∂n = 0 on ∂Ω. (73b) The exact solution for this problem in polar coordinates is [101]: 

u(r) = (1 − r2)2

64 . (74) Given that both u and ∂u/∂ n are specified on the boundary, this problem illustrates the use of a di ff erent solution structure than the ones considered until now. To impose the boundary conditions, we first create a distance function 40 0 5000 10000 Epochs 1.5 × 10 0

1.6 × 10 0

1.7 × 10 0

1.8 × 10 0

1.9 × 10 0

2 × 10 0

> Training loss

REQ MVP 0 5000 10000 Epochs 10 −1

3 × 10 −2

4 × 10 −2

6 × 10 −2

> Error

REQ MVP 0.00 0.25 0.50 0.75 1.00 0.00.20.40.60.81.00.00 0.25 0.50 0.75 1.00 0.00.20.40.60.81.00.00 0.25 0.50 0.75 1.00 0.00.20.40.60.81.00.00 0.15 0.30 0.45 0.60 0.75 0.90 

−0.048 

−0.036 

−0.024 

−0.012 0.000 0.012 0.024 

−0.027 

−0.018 

−0.009 0.000 0.009 0.018 0.027 (a) (b) 

(c) (d) (e) 

Figure 40: Computation of harmonic coordinates (vertex at (0 , 0)) on an L-shaped polygon. (a), (b) Training loss and normalized absolute error for ˜ubc nn (x; θ) (REQ and MVP). (c) Reference finite element solution. Contour plots of the error for (d) ˜ ubc nn (x) (REQ) and (e) ˜ ubc nn (x) (MVP). 0 5000 10000 Epochs 10 1

8 × 10 0

9 × 10 0

> Training loss

REQ MVP 0 5000 10000 Epochs 4 × 10 −1

5 × 10 −1

> Error

REQ MVP 0.00 0.25 0.50 0.75 1.00 0.00.20.40.60.81.00.00 0.25 0.50 0.75 1.00 0.00.20.40.60.81.00.00 0.25 0.50 0.75 1.00 0.00.20.40.60.81.00.00 0.15 0.30 0.45 0.60 0.75 0.90 0.00 0.03 0.06 0.09 0.12 0.15 0.18 0.21 0.24 0.00 0.03 0.06 0.09 0.12 0.15 0.18 0.21 

(a) (b) 

(c) (d) (e) 

Figure 41: Computation of harmonic coordinates (vertex at (1 /2, 1/2)) on an L-shaped polygon. (a), (b) Training loss and normalized absolute error for ˜ ubc nn (x; θ) (REQ and MVP). (c) Reference finite element solution. Contour plots of the error for (d) ˜ ubc nn (x) (REQ) and (e) ˜ ubc nn (x) (MVP). 

φ to the circular boundary. In this case, the distance function can be exactly determined as φ(x) = 1 − √x2 + y2;however, this exact distance function has derivative singularities at the origin. Therefore, we use an approximate 41 distance function to a unit circle that is given in (7), which is reproduced below: 

φ(x) = 1 − x · x

2 ,

which is a bivariate polynomial. Now we construct an ansatz that satisfies both essential boundary conditions us-ing (50): ˜ ubc nn (x; θ) = φ2 ˜u(x; θ). As with previous problems, this problem can be solved using either the collocation approach (see Guo et al. [102]) or the Ritz method. For collocation, we minimize the loss function 

Lbc nn (θ) = ||∇ 4 ˜ubc nn (x; θ) − 1|| 2 

> Ω,NI

,

whereas for the Ritz approach, we minimize the loss function that is presented in Section 7.2.2. Numerical results for collocation and Ritz using ˜ ubc nn (x; θ) are presented in Fig. 42. In the computations, 2800 interior points are used for both methods. For both collocation and Ritz we use the cubic ReLU activation function. Note that for collocation with standard PINN one cannot use the cubic ReLU since it is a biharmonic function. The trial function ˜ ubc nn (x; θ) has other terms that are present in it, and hence in general it is not biharmonic ( ∇4 ˜ubc nn , 0). A representative mesh that display the interior points over a disk is shown in Fig. 15. The network architecture is 2–50–50–1. The training loss as a function of epochs for the collocation and Ritz methods are shown in Fig. 42a. The exact solution is shown in Fig. 42b, and the collocation and Ritz solution for ˜ ubc nn (x) after 10,000 epochs are plotted in Figs 42c and 42d. The surface plots of the error in the Ritz and collocation solutions are presented in Figs 42e and 42f. We observe that the unnormalized error in the Ritz and collocation methods are O(10 −5); the latter is consistent with the accuracy reported in Guo et al. [102]. 0 2500 5000 7500 10000 Epochs 10 −3     

> 10 −2
> 10 −1
> 10 0
> Error Ritz Collocation
> −101−1010.000 0.005 0.010 0.015
> −101−1010.000 0.005 0.010 0.015
> −101−1010.000 0.005 0.010 0.015
> −101−101
> ×10 −5
> −202
> −101−101
> ×10 −5
> −1.0
> −0.50.0

(a) (b) (c) (d) (e) (f)   

> Figure 42: Numerical solutions for clamped circular Kirchho ff plate bending problem using ˜ ubc nn (x;θ) (Ritz and collocation). Network architecture used is 2–50–50–1. (a) Normalized absolute error in training. Surface plots of of the (b) exact solution, and (c), (d) Ritz and collocation solutions. Surface plots of (e) error in Ritz solution and (f) error in collocation solution.

10.5. Eikonal equation 

We consider the Eikonal equation, which is a first-order nonlinear hyperbolic PDE. The boundary-value problem of the Eikonal equation is: 

||∇ u|| = 1

f in Ω ⊂ R2, (75a) 

u = 0 on Γ, (75b) 42 where Γ is an interface in two dimensions and f (x) > 0 is the speed on the interface. If f (x) = 1, then u(x) is the shortest (signed) distance from x to the boundary ∂Ω. If the zero level curve of u(x) represents the initial location of the interface, then u−1(t) yields the location of the interface at time t. Hence, u(x) represents the shortest time (arrival time) that is required to travel from the boundary Γ to x. For monotonically advancing fronts ( f > 0), the fast marching method [72] and the fast sweeping method [103] are highly e ffi cient methods to solve (75). When upwind finite-di ff erences are used to solve (75), it implies a causality: value of u(x) only depends on values of u(y) for which 

u(x) > u(y). No such restriction is used in PINN during training—we use a collocation method, where the exact satisfaction of the Dirichlet condition on Γ is met by constructing ADFs that use R-equivalence from Section 3.2 and the generalized mean value potential from Section 4.1. We solve (75) with f = 1 to compute the signed distance function using PINN. The cubic ReLU activation function, which is a C2 function, is used for all problems. The closed interface Γ is embedded within the biunit square, Ω0 = (−1, 1) 2, and as benchmark problems we consider a ffi ne (polygonal) and curved interfaces for Γ. The first problem that we consider is the computation of the signed distance function to the boundary of a smaller square, 

Ω = (−1/2, 1/2) 2. The Dirichlet boundary condition u = 0 is imposed on Γ = ∂Ω. The network architecture used is 2–30–30–30–1 for ˜ ubc nn and 2–70–70–1 for ˜ unn . For collocation, 10,000 points are used in the interior of the biunit square, and 400 points on the boundary Γ for ˜ unn (x; θ). Numerical results are presented in Fig. 43. The training loss and normalized absolute errors as a function of epochs for ˜ ubc nn (x; θ) (REQ and MVP) and ˜ unn (x; θ) are shown in Figs. 43a and 43b. Once again, we notice that while ˜ unn (x; θ) attains the lowest loss among the three schemes, it has the highest error. This problem has an exact solution, which is shown as a contour plot in Fig. 43c. The exact distance, 

u(x), achieves its maximum value of 1 / √2 at the corners of the biunit square and its minimum value of −1/2 at the center. The error in the numerical solutions are plotted in Figs. 43d–43f. While ˜ ubc nn (x) (REQ and MVP) satisfy the boundary condition exactly, MVP results in a slightly more accurate solution in the entire domain. The L∞ norm of the error using REQ, MVP and ˜ unn (x) are 0.03, 0.026, and 1.34, respectively. The standard PINN, ˜ unn (x; θ), does poorly on satisfying the boundary condition, which in turn leads to larger pointwise errors over the whole domain. 0 20000 40000 Epochs 10 −2              

> 10 −1
> 10 0
> Training loss ˜ubc nn (REQ) ˜ubc nn (MVP) ˜unn
> 020000 40000 Epochs 10 −1
> 10 0
> Error  ˜ubc nn (REQ) ˜ubc nn (MVP) ˜unn
> −101
> −1.0
> −0.50.00.51.0
> −101
> −1.0
> −0.50.00.51.0
> −101
> −1.0
> −0.50.00.51.0
> −101
> −1.0
> −0.50.00.51.0
> −0.027
> −0.018
> −0.009 0.000 0.009 0.018 0.027 0.036
> −0.024
> −0.018
> −0.012
> −0.006 0.000 0.006 0.012 0.018
> −1.12
> −0.96
> −0.80
> −0.64
> −0.48
> −0.32
> −0.16 0.00 0.16

(a) (b) (c) (d) (e) (f)      

> Figure 43: Solving the Eikonal equation using ˜ ubc nn (x;θ) and ˜ unn (x;θ) to compute the signed distance to the boundary of a square. The network architecture used is 2–30–30–30–1 for ˜ ubc nn and 2–70–70–1 for ˜ unn . (a), (b) Training loss and normalized absolute error as a function of epochs for ˜ubc nn (x;θ) (REQ and MVP) and ˜ unn (x;θ). (c) Exact signed distance function. Contour plots of the error for (d) ˜ ubc nn (x) (REQ), (e) ˜ ubc nn (x) (MVP), and (f) ˜ unn (x).

As the next problem, we consider the signed distance function to the boundary Γ of an ellipse that is centered at the origin and with semi-major and semi-minor axes of 0.25 and 0.15, respectively. The approximate distance function to the ellipse, φ(x), is computed using (8) (see Fig. 4b) for REQ and using (16) for MVP (see Fig. 13a) for MVP. The network architecture used is 2–50–50–1. The results that ˜ unn (x) produced are very poor, and hence are not included. Numerical results using ˜ ubc nn (x; θ) (REQ and MVP) are presented in Fig. 44. The training loss and the normalized error as a function of epochs is shown in Figs. 44a and 44b. The exact distance function (computed numerically) is presented in Fig. 44c, and the error plots for the numerical solutions ˜ ubc nn (x) (REQ and MVP) are shown in Figs. 44d 43 and 44e, respectively. Larger errors are concentrated in the region that is close to the center of the ellipse; away from the center the errors are less than 1 percent. The exact distance function is C0 (derivative discontinuities at the center of the ellipse), whereas the numerical solution is C2 smooth. 0 20000 40000 Epochs 10 −2                  

> 10 −1
> 10 0
> Training loss ˜ubc nn (REQ) ˜ubc nn (MVP) 020000 40000 Epochs 10 −2
> 10 −1
> 10 0
> Error ˜ubc nn (REQ) ˜ubc nn (MVP)
> −1.0−0.50.00.51.0
> −1.0
> −0.50.00.51.0
> −1.0−0.50.00.51.0
> −1.0
> −0.50.00.51.0
> −1.0−0.50.00.51.0
> −1.0
> −0.50.00.51.0
> −0.24 0.00 0.24 0.48 0.72 0.96 1.20
> −0.018 0.000 0.018 0.036 0.054 0.072 0.090
> −0.018 0.000 0.018 0.036 0.054 0.072 0.090

(a) (b) (c) (d) (e)   

> Figure 44: Solving the Eikonal equation using ˜ ubc nn (x;θ) to compute the signed distance to the boundary of an ellipse. The network architecture used in 2–50–50–1. (a), (b) Training loss and normalized absolute error in training as a function of epochs for ˜ ubc nn (x;θ) (REQ and MVP) . (c) Exact signed distance function. Contour plots of the error for (d) ˜ ubc nn (x) (REQ) and (e) ˜ ubc nn (x) (MVP).

Lastly, we consider the signed distance function to the boundary Γ of the polygonalized map of Bhutan; see plots of the approximate distance functions to Γ using REQ and MVP that are presented in Figs. 8 and 11, respectively. The network architecture used is 2–50–50–1. Here too the results of ˜ unn (x) are not included since they are very poor. Numerical results using ˜ ubc nn (x; θ) (REQ and MVP) are presented in Fig. 45. The training loss and the normalized absolute error as a function of epochs are shown in Figs 45a and 45b. At 20,000 epochs, the absolute normalized error is O(10 −1). The exact distance function is presented in Fig. 45c, and the error plots for the numerical solutions ˜ ubc nn (x)(REQ and MVP) are presented in Figs. 45d and 45e, respectively. The maximum error is about 4 percent. 

11. Poisson Problem over the Four-Dimensional Hypercube 

As the last problem, we consider the model isotropic steady-state heat conduction (Poisson) problem over the 4-dimensional hypercube to show that essential boundary conditions can be readily imposed in higher dimensions as well. Consider the following Poisson problem with homogeneous Dirichlet boundary conditions: 

−∇ 2u = f in Ω = (−1, 1) 4 (76a) 

u = 0 on ∂Ω, (76b) where u(x) : R4 → R is sought and f (x) is the forcing function. Let x := (x1, x2, x3, x4) ∈ Ω denote a point in the hypercube. We choose f (x) = ∏4 

> i=1

sin( πxi), so that the exact solution is: 

u(x) =

∏4 

> i=1

sin( πxi)4π2 .

As noted in prior high-dimensional studies using PINN [8, 9], numerical solutions for problems in high-dimensions are challenging since there is no easy way to mesh the domain and they are also subject to the curse of dimensionality. 44 0 10000 20000 Epochs 10 −1                  

> 10 0
> Training loss ˜ubc nn (REQ) ˜ubc nn (MVP) 010000 20000 Epochs 10 −1
> 10 0
> Error ˜ubc nn (REQ) ˜ubc nn (MVP)
> −1.0−0.50.00.51.0
> −1.0
> −0.50.00.51.0
> −1.0−0.50.00.51.0
> −1.0
> −0.50.00.51.0
> −1.0−0.50.00.51.0
> −1.0
> −0.50.00.51.0
> −0.32
> −0.16 0.00 0.16 0.32 0.48 0.64 0.80 0.96
> −0.045
> −0.030
> −0.015 0.000 0.015 0.030 0.045
> −0.05
> −0.04
> −0.03
> −0.02
> −0.01 0.00 0.01 0.02 0.03 0.04

(a) (b) 

(c) (d) (e)   

> Figure 45: Solving the Eikonal equation using ˜ ubc nn (x;θ) to compute the signed distance to the boundary of the polygonal map of Bhutan. The network architecture is 2–50–50–1. (a), (b) Training loss and normalized absolute error in training as a function of epochs for ˜ ubc nn (x;θ) (REQ and MVP). (c) Exact signed distance function. Contour plots of the error for (d) ˜ ubc nn (x) (REQ) and (e) ˜ ubc nn (x) (MVP).

Among meshfree methods, since construction of radial basis functions is dimension-independent, RBF-based mesh-free methods have had success in solving high-dimensional problems [104]. It is in these problems that the power and potential of a meshfree method such as PINN becomes most apparent. For this problem, we only consider trial func-tions, ˜ ubc nn (x; θ), which exactly enforce the homogeneous Dirichlet boundary condition on ∂Ω. One way of enforcing the boundary conditions is to assume ˜ ubc nn to be of the form ˜ubc nn (x; θ) =



> 4

∏

> i=1

(1 − x2 

> i

)

 ˜uR

> nn

(x; θ),

which we refer to as the ‘product method’ and note that while this is an obvious approach for the present problem, it leads to very small numbers inside the domain and away from the boundaries. In this case, the multiplicative factor scales as O(x8) inside the biunit hypercube, and therefore the network parameters have to compensate for this highly nonlinear behavior during training. It is preferable to have a multiplicative factor that is better behaved in order to aid the training. To this end, we construct φ(x) using R-equivalence in (10), which seamlessly extends to higher dimensions. For this choice, the trial function ˜ ubc nn (x; θ) is of the form: ˜ubc nn (x; θ) = φ(x) ˜ uR

> nn

(x; θ), (77) where φ(x) consists of R-equivalence (REQ) operations on φi(x), where φi(x) is the R-function for the region (strip) bounded by the hyperplanes 1 − xi and xi − 1. We form φi(x) = (1 − x2 

> i

)/2, which is an ADF that is normalized to order 1. On using the REQ composition in (10), we write 

φ(x) = φ1(x) ∼ φ2(x) ∼ φ3(x) ∼ φ4(x), (78) which generalizes to the hypercube in Rd. Note that φ(x) only scales as x2 

> i

in each coordinate direction, and therefore is much better behaved. For this problem, another choice for φ(x) is to define two φi’s in each dimension, i.e., 

φ2i−1 = 1 + xi and φ2i = 1 − xi and then define φ(x) = φ1 ∼ φ2 . . . φ 8 for the 4-dimensional hypercube. For m = 1, this construction coincides with the expression for φ in (78). In Fig. 46, we present the numerical solutions using the product and REQ ( m = 1) forms of the trial function. For both choices, we consider 5,000 randomly generated interior points in Ω for training and compute the normalized 45 error at a separate set of 5,000 interior points. The network architecture for both choices is 4–100–100–1. The isosurface plot of φ(x) ( x4 = 0 plane) using REQ is shown in Fig. 46a. In Fig. 46b, the evolution of the training loss and normalized absolute error is presented for the product and REQ trial functions. We observe that while the REQ method is able to reach error levels of about 1 percent, the product method does not converge. The ˜ ubc nn (x) solution with REQ yields O(10 −2) error, whereas the product method has O(1) error even though it has a much smaller PDE loss of O(10 −5).           

> (a) 010000 20000 Epochs 10 −5
> 10 −4
> 10 −3
> 10 −2
> 10 −1
> Training loss REQ Product 010000 20000 Epochs 10 −1
> 10 0
> Error  REQ Product
> (b)
> Figure 46: Numerical solution of the Poisson problem over the four-dimensional hypercube. The network architecture is 2–100–100–1. The trial function, ˜ ubc nn (x), is constructed using R-equivalence (REQ) and the product method. (a) Isosurface plot of the approximate distance function using REQ, φ(x), over the 3-dimensional biunit cube ( x4=0), with φ=0 being satisfied on the boundary of the cube. (b) Training loss and normalized absolute error for REQ and the product method as a function of epochs.

12. Conclusions 

Starting from the seminal works of Lagaris et al. [1–3] and the recent extensions and major advancements by Raissi et al. [6] and E and Yu [9], there has been a surge in the development and application of physics-informed neural networks to solve partial di ff erent equations. In this paper, we have introduced a new approach based on distance fields to construct geometry-aware approximations in physics-informed neural networks by ensuring that the necessary boundary conditions are met a priori: all boundary conditions in a collocation method, and the essential boundary conditions (kinematic admissibility) in a Ritz method. Our approach relied on the theory of R-functions [23, 24] to construct approximate distance fields and their use within a meshfree method to exactly impose boundary conditions to solve PDEs [27, 30]. Apart from R-functions, we also showed that mean value potential fields [34–36] can be used to construct suitable distance field to solve PDEs over domains with a ffi ne as well as curved boundaries. We presented several numerical examples to reveal the benefits of exactly imposing the boundary conditions versus the current state-of-the-art in deep collocation and deep Ritz methods for physics-informed neural networks. Notably, requiring only the interior residual error contribution in the loss function simplifies the training of the network and leads to more accurate numerical solutions. This was shown through several verification tests on benchmark one- and two-dimensional boundary-value problems—and consistently revealed the pitfalls of being guided by the magnitude of the loss function in standard PINN-based collocation approaches [6]. In PINN methods, when there are multiple terms (PDE loss and boundary losses) that are present in the loss function and the loss weights are fixed a priori, the magnitude of the loss function at the end of the training does not provide a measure of the accuracy of the approx-imation. There can be a many-fold di ff erence in the two error measures, which is revealed in our simulations. This is not surprising since the weights associated with each loss term is not known a priori since it depends on the PDE, boundary conditions, and the training. There is no clear rationale way to set these weights in order to ensure that the approach is robust and guaranteed to lead to reliable results. This study has reinforced that it is important that the PDE loss stands on its own, and boundary conditions are enforced via the ansatz. One approach is to construct a separate neural network to meet the boundary conditions as is done by Berg and Nystr¨ om [7], but the inherent inaccuracy in the satisfaction of the boundary conditions then propagates when training for the PDE is conducted. Moreover, when the geometry is complicated and di ff erent types of boundary conditions are imposed on di ff erent subset of the boundary, 46 then this approach may soon become impractical. Our approach ensures exact satisfaction of all necessary boundary condition, which makes it appealing—so that training of the network (training loss and PDE loss coincide) is more effi cient and accurate solutions can be realized. This study has provided a method to perform meshfree analysis—solving PDEs without domain discretization— on complex two-dimensional geometries using physics-informed neural networks. This was made possible on using approximate distance functions, which were based on R-functions (R-equivalence composition) and generalized mean value potentials, in conjunction with transfinite inverse-distance based interpolation to exactly satisfy the boundary conditions. For the problems that we considered, both ADFs delivered the same order of accuracy with the constant being smaller for R-equivalence in most cases (Eikonal equation was the exception). If u = 0 is imposed on the bound-ary of a polygon with many (tens to hundreds) edges, then the ADF using mean value potential is more e ffi cient since it results in an ADF by construction, whereas R-equivalence requires first constructing an ADF to each edge and then the joining operation to be performed. However, if di ff erent boundary conditions are imposed on distinct boundary segments then only R-equivalence is viable since this cannot be realized using the ADF based on mean value poten-tial fields. While we have used R-functions and the mean value potential to construct smooth approximate distance functions, fields such as constructive geometric modeling [105] with implicit functions [73, 106–108], PDE-based solutions for distance computations [109–111], and deep learning in computer vision [112] are rapidly advancing and may o ff er other attractive alternatives to construct smooth distance fields for use in deep neural networks to solve PDEs. A separate and more in depth investigation is needed to explore if there are other network architectures and optimizers for network training that are better-suited for the PINN ansatz with approximate distance functions, and to also quantify the accuracy that is obtained using di ff erent ADFs. Lastly, the formulation permitted exact model-ing of a ffi ne and curved boundaries, thereby providing a pathway to conducting simulations on the exact geometry (isogeometric analysis) [37]. The ideas herein can be extended for higher dimensional problems, since R-equivalence composition for the ap-proximate distance function is additive and does not su ff er from the curse of dimensionality. This was demonstrated in Section 11, where we obtained an accurate PINN solution for a Poisson problem over the 4-dimensional hypercube. Extending our formulation to complex geometries in 3D, and the development of a deep Petrov-Galerkin domain-decomposition method are topics that we plan to pursue. 

13. Acknowledgments 

AS acknowledges support from the NSF CAREER grant #1554033 to the Illinois Institute of Technology. NS thanks Anand Reddy, Eric Chin and Kai Hormann for many helpful discussions. 

References             

> [1] I. E. Lagaris, A. Likas, D. I. Fotiadis, Artifical neural networks for solving ordinary and partial di ff erential equations, IEEE Transactions on Neural Networks 9 (5) (1998) 987–1000. [2] I. E. Lagaris, A. Likas, D. I. Fotiadis, Artifical neural network methods in quantum mechanics, Computer Physics Communications 104 (1997) 1–14. [3] I. E. Lagaris, A. C. Likas, D. G. Papageorgiou, Neural-network methods for boundary value problems with irregular boundaries, IEEE Transactions on Neural Networks 11 (5) (2000) 1041–1049. [4] K. S. McFall, An artificial neural network method for solving boundary value problems with arbitrary irregular boundaries, Ph.D. thesis, Georgia Institute of Technology, Atlanta, GA, USA (2006). [5] K. S. McFall, J. R. Mahan, Artificial neural network method for solution of boundary value problems with exact satisfaction of arbitrary boundary conditions, IEEE Transactions on Neural Networks 20 (8) (2009) 1221–1233. [6] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for forward and inverse problems involving nonlinear partial di ff erential equations, Journal of Computational Physics 378 (2019) 686–707. [7] J. Berg, K. Nystr¨ om, A unified deep artificial neural network approach to partial di ff erential equations in complex geometries, Neuralcom-puting 317 (2018) 28–41. [8] J. Sirignano, K. Spiliopoulos, DGM: A deep learning algorithm for solving partial di ff erential equations, Journal of Computational Physics 375 (2018) 1339–1364. [9] W. E, B. Yu, The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems, Communications in Mathematics and Statistics 6 (1) (2018) 1–12. [10] J. Han, A. Jentzen, W. E, Solving high-dimensional partial di ff erential equations using deep learning, Proceedings of the National Academy of Sciences 115 (34) (2018) 8505–8510. [11] E. Kharazmi, Z. Zhang, G. E. Karniadakis, Variational physics-informed neural networks for solving partial di ff erential equations (2019).
> arXiv:1912.00873 .[12] E. Kharazmi, Z. Zhang, G. E. Karniadakis, hp-VPINNs: Variational physics-informed neural networks with domain decomposition, Com-puter Methods in Applied Mechanics and Engineering 374 (2020) 113547.

47 [13] L. Lu, X. Meng, Z. Mao, G. E. Karniadakis, DeepXDE: A deep learning library for solving di ff erential equations, SIAM Review 63 (1) (2021) 208–228. [14] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, et al., Tensorflow: A system for large-scale machine learning, in: 12th {USENIX } symposium on operating systems design and implementation ( {OSDI } 16), 2016, pp. 265–283. [15] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, Pytorch: An imperative style, high-performance deep learning library, in: H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alch´ e Buc, E. Fox, R. Garnett (Eds.), Advances in Neural Information Processing Systems, Vol. 32, Curran Associates, Inc., 2019. [16] S. Wang, Y. Teng, P. Perdikaris, Understanding and mitigating gradient pathologies in physics-informed neural networks (2020). arXiv: 2001.04536 .[17] J. Chen, R. Du, K. Wu, A comparison study of deep Galerkin method and deep Ritz method for elliptic problems with di ff erent boundary conditions, Communications in Mathematical Research 36 (3) (2020) 354–376. [18] L. Lyu, K. Wu, R. Du, J. Chen, Enforcing exact boundary and initial conditions in the deep mixed residual method (2020). arXiv: 2008.01491 .[19] I. Babuˇ ska, U. Banerjee, J. E. Osborn, Survey of meshless and generalized finite element methods: a unified approach, Acta Numerica 12 (2003) 1–125. [20] A. Huerta, T. Belytschko, S. Fern´ andez-M´ endez, T. Rabczuk, X. Zhuang, M. Arroyo, Meshfree methods, 2nd Edition, Vol. 2 of Encyclopedia of Computational Mechanics, Wiley, 2017, Ch. 3, pp. 1–38. [21] L. V. Kantorovich, V. I. Krylov, Approximate Methods of Higher Analysis, Interscience, New York, NY, USA, 1958. [22] V. L. Rvachev, Theory of R-functions and Some Applications, Naukova Dumka, Kiev. In Russian, 1982. [23] V. L. Rvachev, T. I. Sheiko, R-functions in boundary value problems in mechanics, Applied Mechanics Reviews 48 (4) (1995) 151–188. [24] V. L. Rvachev, T. I. Sheiko, V. Shapiro, I. Tsukanov, On completeness of RFM solution structures, Computational Mechanics 25 (2000) 305–3163. [25] V. L. Rvachev, T. I. Sheiko, V. Shapiro, I. Tsukanov, Transfinite interpolation over implicitly defined sets, Computer Aided Geometric Design 18 (2001) 195–220. [26] V. Shapiro, Theory of R-functions and applications: A primer, Tech. Rep. CPA88-3, Cornell Programmable Automation, Sibley School of Mechanical Engineering, Ithaca, NY 14853, USA (1991). [27] V. Shapiro, I. Tsukanov, Meshfree simulation of deforming domains, Computer-Aided Design 31 (7) (1999) 459–471. [28] V. Shapiro, I. Tsukanov, The architecture of SAGE–a meshfree system based on RFM, Engineering with Computers 18 (4) (2002) 295–311. [29] A. Biswas, V. Shapiro, Approximate distance fields with non-vanishing gradients, Graphical Models 66 (3) (2004) 133–159. [30] V. Shapiro, Semi-analytic geometry with R-functions, Acta Numerica 16 (2007) 239–303. [31] M. Freytag, V. Shapiro, I. Tsukanov, Finite element analysis in situ, Finite Elements in Analysis and Design 47 (9) (2011) 957–972. [32] K. H¨ ollig, U. Reif, J. Wipper, Weighted extended B-spline approximation of Dirichlet problems, SIAM Journal on Numerical Analysis 39 (2) (2001) 442–462. [33] D. Mill´ an, N. Sukumar, M. Arroyo, Cell-based maximum-entropy approximants, Computer Methods in Applied Mechanics and Engineering 284 (2015) 712–731. [34] M. S. Floater, Mean value coordinates, Computer Aided Geometric Design 20 (1) (2003) 19–27. [35] C. Dyken, M. S. Floater, Transfinite mean value interpolation, Computer Aided Geometric Design 26 (1) (2009) 117–134. [36] A. Belyaev, P.-A. Fayolle, A. Pasko, Signed Lp-distance fields, Computer-Aided Design 45 (2) (2013) 523–528. [37] T. J. R. Hughes, J. A. Cottrell, Y. Bazilevs, Isogeometric analysis: CAD, finite elements, NURBS, exact geometry and mesh refinement, Computer Methods in Applied Mechanics and Engineering 194 (39–41) (2005) 4135–4195. [38] P. Thoutireddy, M. Ortiz, A variational r-adaption and shape-optimization method for finite-deformation elasticity, International Journal for Numerical Methods in Engineering 61 (1) (2004) 1–21. [39] J. He, L. Li, J. Xu, C. Zheng, ReLU deep neural networks and linear finite elements (2018). arXiv:1807.03973 .[40] E. Grinspun, The basis refinement method, Ph.D. thesis, California Institute of Technology, Pasadena, CA, USA (2003). [41] E. C. Cyr, M. A. Gulian, R. G. Patel, M. Perego, N. A. Trask, Robust training and initialization of deep neural networks: An adaptive basis viewpoint, in: Mathematical and Scientific Machine Learning, PMLR, 2020, pp. 512–536. [42] J. A. A. Opschoor, P. C. Petersen, C. Schwab, Deep ReLU networks and high-order finite element methods, Analysis and Applications 18 (05) (2020) 715–770. [43] E. J. Kansa, Multiquadrics—A scattered data approximation scheme for applications to computational fluid-dynamics. 1. Surface approxi-mations and partial derivative estimates, Computers & Mathematics with Applications 19 (8 /9) (1990) 127–145. [44] E. J. Kansa, Multiquadrics—A scattered data approximation scheme for applications to computational fluid-dynamics. 2. Solutions to parabolic, hyperboloc and elliptic partial-di ff erential equations, Computers & Mathematics with Applications 19 (8 /9) (1990) 147–161. [45] M. D. Buhmann, Radial basis functions: theory and implementations, Cambridge University Press, Cambridge, UK, 2003. [46] G. Fasshauer, Meshfree Approximation Methods in MATLAB, Interdisciplinary Mathematical Sciences – Vol. 6, World Scientific Publishers, Singapore, 2007. [47] R. Schaback, H. Wendland, Kernel techniques: from machine learning to meshless methods, Acta Numerica 15 (2006) 543. [48] M. Arroyo, M. Ortiz, Local maximum-entropy approximation schemes: a seamless bridge between finite elements and meshfree methods, International Journal for Numerical Methods in Engineering 65 (13) (2006) 2167–2202. [49] I. Babuˇ ska, J. M. Melenk, The partition of unity method, International Journal for Numerical Methods in Engineering 40 (1997) 727–758. [50] V. Rajan, Optimality of the Delaunay triangulation in Rd , Discrete & Computational Geometry 12 (1) (1994) 189–202. [51] N. Sukumar, Construction of polygonal interpolants: a maximum entropy approach, International Journal for Numerical Methods in Engi-neering 61 (12) (2004) 2159–2181. [52] N. Sukumar, Maximum entropy approximation, AIP Conference Proceedings 803 (1) (2005) 337–344. [53] M. Arroyo, M. Ortiz, Local maximum-entropy approximation schemes, in: M. Griebel, M. A. Schweitzer (Eds.), Meshfree Methods for Partial Di ff erential Equations III, Vol. 57 of Lecture Notes in Computational Science and Engineering, Springer, Berlin, Germany, 2007, pp. 1–16. 

48 [54] N. Sukumar, R. W. Wright, Overview and construction of meshfree basis functions: From moving least squares to entropy approximants, International Journal for Numerical Methods in Engineering 70 (2) (2007) 181–205. [55] A. Rosolen, D. Mill´ an, M. Arroyo, On the optimum support size in meshfree methods: a variational adaptivity approach with maximum entropy approximants, International Journal for Numerical Methods in Engineering 82 (7) (2010) 868–895. [56] J. Park, I. W. Sandberg, Universal approximation using radial-basis-function networks, Neural computation 3 (1991) 246–257. [57] H. N. Mhaskar, Neural networks for optimal approximation of smooth and analytic functions, Neural computation 8 (1) (1996) 164–177. [58] K. Lee, N. A. Trask, R. G. Patel, M. A. Gulian, E. C. Cyr, Partition of unity networks: deep hp-approximation (2021). arXiv:2101. 11256 .[59] A. A. Ramabathiran, P. Ramachandran, SPINN: Sparse, physics-based, and partially interpretable neural networks for PDEs, Journal of Computational Physics 445 (2021) 110600. [60] F. Greco, M. Arroyo, High-order maximum-entropy collocation methods, Computer Methods in Applied Mechanics and Engineering 367 (2020) 113115. [61] H. Sheng, C. Yang, PFNN: A penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries, Journal of Computational Physics 428 (2021) 110085. [62] V. Dwivedi, B. Srinivasan, Physics informed extreme learning machine (PIELM)–A rapid method for the numerical solution of partial di ff erential equations, Neuralcomputing 391 (2020) 96–118. [63] V. Dwivedi, B. Srinivasan, Solution of biharmonic equation in complicated geometries with physics informed extreme learning machine, Journal of Computing and Information Science in Engineering 20 (6) (2020). [64] Y. Liao, P. Ming, Deep Nitsche method: Deep Ritz method with essential boundary conditions (2019). arXiv:1912.01309 .[65] S. Li, W. K. Liu, Meshfree Particle Methods, Springer-Verlag, New York, NY, USA, 2004. [66] K. Hornik, M. Stinchcombe, H. White, Multilayer feedforward networks are universal approximators, Neural Networks 2 (1989) 359–366. [67] K. Hornik, Approximation capabilities of multilayer perceptrons, Neural Networks 4 (1991) 251–257. [68] G. Strang, G. J. Fix, An Analysis of the Finite Element Method, Prentice–Hall, New York, NY, USA, 1973. [69] F. M. Rohrhofer, S. Posch, B. C. Geiger, On the pareto front of physics-informed neural networks (2021). arXiv:2105.00862 .[70] O. Hennigh, S. Narasimhan, M. A. Nabian, A. Subramaniam, K. Tangsali, Z. Fang, M. Rietmann, W. Byeon, S. Choudhry, NVIDIA SimNet ™: An AI-accelerated multi-physics simulation framework, in: International Conference on Computational Science, Springer, 2021, pp. 447–461. [71] I. Tsukanov, S. R. Posireddy, Hybrid method of engineering analysis: Combining meshfree method with distance fields and collocation technique, Journal of Computing and Information Science in Engineering 11 (3) (2011). [72] J. A. Sethian, Level Set Methods and Fast Marching Methods: Evolving Interfaces in Computational Geometry, Fluid Mechanics, Computer Vision, and Materials Science, Cambridge University Press, Cambridge, U.K., 1999. [73] J. Bloomenthal, Bulge elimination in convolution surfaces, Computer Graphics Forum 16 (1) (1997) 31–41. [74] V. Shapiro, I. Tsukanov, Implicit functions with guaranteed di ff erential properties, in: Proceedings of the Fifth ACM Symposium on Solid Modeling and Applications, 1999, pp. 258–269. [75] K. Upreti, T. Song, A. Tambat, G. Subbarayan, Algebraic distance estimations for enriched isogeometric analysis, Computer Methods in Applied Mechanics and Engineering 280 (2014) 28–56. [76] E. B. Chin, N. Sukumar, Modeling curved interfaces without element-partitioning in the extended finite element method, International Journal for Numerical Methods in Engineering 120 (5) (2019) 607–649. [77] A. G. Belyaev, P.-A. Fayolle, Transfinite barycentric coordinates, in: Hormann and Sukumar [80], pp. 43–62. [78] M. S. Floater, Generalized barycentric coordinates and applications, Acta Numerica 24 (2015) 161–214. [79] D. Anisimov, Barycentric coordinates and their properties, in: Hormann and Sukumar [80], pp. 3–22. [80] K. Hormann, N. Sukumar (Eds.), Generalized Barycentric Coordinates in Computer Graphics and Computational Mechanics, CRC Press, New York, NY, 2017. [81] K. Hormann, M. S. Floater, Mean value coordinates for arbitrary planar polygons, ACM Transactions on Graphics 25 (4) (2006) 1424–1441. [82] S. Bruvoll, M. S. Floater, Transfinite mean value interpolation in general dimension, Journal of Computational and Applied Mathematics 233 (7) (2010) 1631–1639. [83] T. Ju, S. Schaefer, J. Warren, Mean value coordinates for closed triangular meshes, ACM Transactions on Graphics 24 (3) (2005) 561–566. [84] E. B. Chin, N. Sukumar, Scaled boundary cubature scheme for numerical integration over planar regions with a ffi ne and curved boundaries, Computer Methods in Applied Mechanics and Engineering 380 (2021) 113796. [85] D. Shepard, A two-dimensional interpolation function for irregularly-spaced data, in: Proceedings of the 23rd ACM national conference, Association for Computing Machinery, New York, New York, 1968, pp. 517–524. [86] F. Rosenblatt, The perceptron: A probabilistic model for information storage and organization in the brain., Psychological Review 65 (6) (1958) 386. [87] Y. LeCun, The MNIST database of handwritten digits, http: // yann.lecun.com /exdb /mnist / (1998). [88] D. Finol, Y. Lu, V. Mahadevan, A. Srivastava, Deep convolutional neural networks for eigenvalue problems in mechanics, International Journal for Numerical Methods in Engineering 118 (5) (2019) 258–275. [89] K. Hornik, Some new results on neural network approximation, Neural Networks 6 (1993) 1069–1072. [90] A. LeNail, Nn-svg: Publication-ready neural network architecture schematics, Journal of Open Source Software 4 (33) (2019) 747. [91] D. P. Kingma, J. Ba, Adam: A method for stochastic optimization (2014). arXiv:1412.6980 .[92] E. Samaniego, C. Anitescu, S. Goswami, V. M. Nguyen-Thanh, H. Guo, K. Hamdia, X. Zhuang, T. Rabczuk, An energy approach to the solution of partial di ff erential equations in computational mechanics via machine learning: Concepts, implementation and applications, Computer Methods in Applied Mechanics and Engineering 362 (2020) 112790. [93] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, Q. Zhang, JAX: composable transformations of Python +NumPy programs URL http://github.com/google/jax (2018). [94] E. Bisong, Google colaboratory, in: Building Machine Learning and Deep Learning Models on Google Cloud Platform, Springer, 2019, pp. 59–64. [95] N. Schl¨ omer, J. Hariharan, dmsh, (2020). Available at https://github.com/nschloe/dmsh . Accessed on April 1, 2021. [96] P.-O. Persson, G. Strang, A simple mesh generator in MATLAB, SIAM Review 46 (2) (2004) 329–345. 

49 [97] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, A. Courville, On the spectral bias of neural networks, in: International Conference on Machine Learning, PMLR, 2019, pp. 5301–5310. [98] S. Wang, H. Wang, P. Perdikaris, On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks, Computer Methods in Applied Mechanics and Engineering 384 (2021) 113938. [99] J. Pushkar, M. Meyer, T. DeRose, B. Green, T. Sanocki, Harmonic coordinates for character articulation, ACM Transactions on Graphics 26 (3) (2007) Article 71. [100] J. R. Shewchuk, Triangle: Engineering a 2D Quality Mesh Generator and Delaunay Triangulator, in: M. C. Lin, D. Manocha (Eds.), Applied Computational Geometry: Towards Geometric Engineering, Vol. 1148 of Lecture Notes in Computer Science, Springer-Verlag, 1996, pp. 203–222. [101] S. P. Timoshenko, S. Woinowsky-Krieger, Theory of Plates and Shells, 2nd Edition, McGraw Hill, New York, NY, USA, 1959. [102] H. Guo, X. Zhuang, T. Rabczuk, A deep collocation method for the bending analysis of Kirchho ff plate (2021). arXiv:2102.02617 .[103] H. Zhao, A fast sweeping method for Eikonal equations, Mathematics of Computation 74 (250) (2005) 603–627. [104] T. Cecil, J. Qian, S. Osher, Numerical methods for high dimensional Hamilton–Jacobi equations using radial basis functions, Journal of Computational Physics 196 (1) (2004) 327–347. [105] A. Ricci, A constructive geometry for computer graphics, The Computer Journal 16 (2) (1973) 157–160. [106] A. Sherstyuk, Kernel functions in convolution surfaces: a comparative analysis, The Visual Computer 15 (4) (1999) 171–182. [107] L. Barthe, B. Wyvill, E. De Groot, Controllable binary CSG operators for “soft objects”, International Journal of Shape Modelling 10 (02) (2004) 135–154. [108] O. Gourmel, L. Barthe, M.-P. Cani, B. Wyvill, A. Bernhardt, M. Paulin, H. Grasberger, A gradient-based implicit blend, ACM Transactions on Graphics 32 (2) (2013) 12:1–12:12. [109] A. Belyaev, P.-A. Fayolle, On variational and PDE-based distance function approximations, Computer Graphics Forum 34 (8) (2015) 104– 118. [110] K. Crane, C. Weischedel, M. Wardetzky, The heat method for distance computation, Communications of the ACM 60 (11) (2017) 90–99. [111] A. G. Belyaev, P.-A. Fayolle, A variational method for accurate distance function estimation, in: Numerical Geometry, Grid Generation and Scientific Computing, Springer, Cham, 2019, pp. 175–181. [112] J. J. Park, P. Florence, J. Straub, R. Newcombe, S. Lovegrove, Deepsdf: Learning continuous signed distance functions for shape represen-tation, in: Proceedings of the IEEE /CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 165–174. 

50
