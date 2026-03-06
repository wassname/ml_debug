Title: 2208.09123v3.pdf

URL Source: https://arxiv.org/pdf/2208.09123

Published Time: Mon, 23 Jan 2023 17:11:52 GMT

Number of Pages: 68

Markdown Content:
# IAN: Iterated Adaptive Neighborhoods for manifold learning and dimensionality estimation 

## Luciano Dyballa 1 and Steven W. Zucker 1,2 

> 1

Department of Computer Science, Yale University, New Haven, CT 

> 2

Department of Biomedical Engineering, Yale University, New Haven, CT 

Abstract 

Invoking the manifold assumption in machine learning requires knowledge of the mani-fold’s geometry and dimension, and theory dictates how many samples are required. However, in most applications the data are limited, sampling may not be uniform, and the manifold’s properties are unknown; this implies that neighborhoods must adapt to the local structure. We introduce an algorithm for inferring adaptive neighborhoods for data given by a similar-ity kernel. Starting with a locally-conservative neighborhood (Gabriel) graph, we sparsify it iteratively according to a weighted counterpart. In each step, a linear program yields minimal neighborhoods globally, and a volumetric statistic reveals neighbor outliers likely to violate manifold geometry. We apply our adaptive neighborhoods to non-linear dimensionality re-duction, geodesic computation, and dimension estimation. A comparison against standard algorithms using, e.g., k-nearest neighbors, demonstrates the usefulness of our approach. 

Research supported by NIH Grant EY031059, NSF CRCNS Grant 1822598, and the Swartz Foundation. This project derived from problems in manifold inference involving neuroscience data. We thank G. Field and M. Stryker for motivating discussions. 

# 1 Introduction 

A starting point for many algorithms in data science—from clustering to manifold inference— is knowing the neighbor relationships among data points. Clustering, for example, often begins with a “ k-nearest neighbor graph,” while manifold inference involves a kernel, i.e., a measure of similarity between data points. In the first case, the neighborhoods are local and discrete; in the second, they are global and continuous, with concentration of influence controlled by the kernel bandwidth, or scale. Such neighbor relationships are fundamental to defining a topology. More-over, dimensionality may be estimated based on the rate of change in the density of points within a ball, that is, within a neighborhood, with respect to its radius. It is helpful when the number of data points is large, a requirement that grows with dimensionality; asymptotic analysis is often favored by theoreticians. 1

> arXiv:2208.09123v3 [cs.LG] 7 Jan 2023

In practice, we rarely have enough data points to satisfy asymptotic bounds. Nor are we given the precise number of neighbors, k, that each point should have. We often make the manifold assumption—that the data points are drawn randomly from a (or near a) manifold—but rarely try to assess the basic properties of the manifold assumed by theorists: its dimensionality, sampling density, curvature, medial axis, or reach (defined in the next section). All of these could influence 

k.Instead, we rely on different visualization algorithms, such as Isomap, diffusion maps, t-SNE, and many others (references in the next section), to find a pleasing organization of the data. This is dangerous, of course, because these algorithms have free parameters. In particular, and central to this paper, most require specifying the number of neighbors, k (or its equivalent): changing k or other parameters changes the result. Unless one knows the answer, one is caught in a conundrum: imposing a prior belief amounts to “fixing” the solution (examples of changing k are shown later in the paper). This gap between theory and practice shows up right from the start. If the manifold is not pure, i.e., if it consists of a union of manifolds of possibly different dimensionality, then there may be no global k that suffices; furthermore, the manifold may have a boundary. Even if it is pure and without boundary, the temptation to choose k large is common. But this can incorrectly fill in the open space around curved manifolds (“folding”, or “short-circuiting”), linking distant points that should not be neighbors. On the other hand, choosing k small can induce holes and break connectivity. Such phenomena are illustrated in Figure 1. As we shall demonstrate, sampling issues and manifold geometry interact in causing these. Moreover, in real datasets the appropriate number of neighbors may differ from point to point. This final issue is a principal motivation for this paper. We present an algorithm to estimate an effective neighborhood—the immediate neighbors, or scale of a similarity kernel—around each point. We seek to identify those nearest neighbors that are “correct” in the sense that they support dimensionality and volume estimates, and manifold inference in general, without covering holes or filling in concavities. It is inspired by the philo-sophical position that views discrete and continuous mathematics as “two sides of the same,” as argued by Lov´ asz [74], and iterates between them. Our algorithm builds from a conservative initial estimate of neighbors (based on a discrete construct, the Gabriel graph) toward a refined one, based on continuous estimates from a multiscale Gaussian kernel. The discrete and continuous volume estimates must be consistent, however, and this provides the glue for our iteration. Since not all of the initial putative neighbors may actually be closest neighbors, those neighbors that violate the volume relationship are pruned, and the process repeats until the two perspectives agree. Our algorithm, thus, can be considered an iterative graph sparsification. Technically, it involves two different graphs: a discrete one, that links only putative nearest neighbors (pairs of points defining the diameter of an otherwise empty ball), and a weighted one, structured by a multiscale Gaussian kernel, whose individual scales must cover the neighborhood given in the discrete graph. Keeping the two graphs consistent is another way to think about our iteration. Each resulting graph can be applied to many different algorithms for data visualization, dimensionality reduction, and manifold inference. Our approach to the problem is in the spirit of exploratory data analysis ; it works with the available data. This provides another view regarding the interaction between sampling and geom-etry: one can only do as well as the available data allow (see Figure 2). The situation is analogous 2mfds Figure 1: Inferring the geometry of manifolds requires neighborhoods around each given data point. Setting the correct scale for these neighborhoods, shown as balls, is fundamental. (A)Example of a 1-dimensional manifold, M. ( B) Collection of points sampled from an unknown distribution over M. Their pairwise distances are the only available data; properties of M are not given a priori . ( C) Using a global kernel scale: if it is too small, the manifold will appear disconnected, artificially producing clusters. Notice how some balls do not touch. (D) If it is too big, the manifold may collapse, giving rise to incorrect geometry/topology. Notice how the balls overlap (covering dimension). (E) The use of local scales based on a global number of nearest neighbors (in this example, k = 2) is still susceptible to the problems above. (F) Our approach computes locally adaptive neighborhood sizes, resulting in scales that conform to the local geometry and sampling. 3Figure 2: Sampling a Swiss cheese: the available data constrain manifold complexity. Viewed from left to right, as the number of sample points increases, the apparent manifold goes from a plane to a plane with holes. The central panel shows the sampling density for which actual holes in the manifold become roughly distinguishable from holes due to sampling. The results of our algorithm on these examples are shown in Figure 23. to that in learning theory, where there is a trade-off between the accuracy of the learner and the coarseness of the hypothesis class over which she is learning [2]. Here, the space of manifolds over which inferences are made is dictated by the available samples. An overview of the paper is as follows. In the next section, we review the background in some detail, covering both the zoo of similarity kernels that exist plus several relevant notions, such as the reach of a manifold, that are well studied in the theory literature. The discussion is organized to emphasize the centrality of scale, or neighborhood, in all of the references. In section 3 we provide an overview of our algorithm. It includes a brief sketch of both graphs we work with, plus the connection back to manifolds. Pseudocode for the algorithm is given in Algorithm 1, which also includes pointers to where each of its steps is developed. We then expand on the algorithm. In section 3.3 we study the Gabriel graph and putative neighbors. Two features are emphasized: scale-free neighborhoods and the relationship between node degree and local dimensionality. A structural criterion is revealed, showing how putative edges between neighbors fill “volumes” that block others from being neighbors. This graph serves as an initialization. It is then refined iteratively in several steps. First, continuous kernel scales are computed based on the discrete, putative neighbors. A linear program relaxation bridges local scales to a global cover, in which each node’s weighted degree is comparable to the number of its neighbors. In other words, each neighborhood radius should not cover too many outside points. If it does, then it indicates that the neighborhood itself should be refined. That is, some putative scales are likely wrong, in the sense that their neighborhood contains an extreme outlier. This leads directly to a volumetric statistic (section 3.5.1), and to a pruning technique for sparsifying edges from the discrete graph. The process iterates until there are no more outliers. In section 4, we evaluate the results for estimating manifold low-dimensional embeddings, geodesics, and local intrinsic dimensionality. Comparisons against popular algorithms, such as UMAP and t-SNE, illustrate the power of the approach. In the end, we demonstrate that it is possi-ble to infer data-driven local neighborhoods that remain consistent with geometric and topological properties of manifolds. Code for our algorithm is available at github.com/dyballa/IAN .42 Background 

Manifold learning is a vast area of machine learning where high-dimensional data are analyzed based on the assumption that they were sampled from a low-dimensional manifold, M [42], in which case geodesic distances over M provide a better description of the relationships between data points than Euclidean distances in ambient space [11]. The manifold assumption finds appli-cations in non-linear dimensionality reduction [102], de-noising [56], interpolation [23], dimen-sionality estimation [25], computational geometry [32], and more. Since M locally resembles Euclidean space, it is standard to define a similarity kernel to define (possibly weighted) neighborhoods around each point in terms of other points. This naturally leads to a graph having data points as nodes and similarity values as edge weights. Then, by computing the graph Laplacian, one can apply a variety of methods from spectral graph theory [see, e.g., 94]. Formal analysis involves the limit as the number of data points grows large; the practical success of such methods depends on how well graph neighborhoods capture the topology and geometry of 

M.We here review the many approaches to specifying a similarity kernel or a local neighborhood. Let M be a d-dimensional manifold in ambient space Rn. When only pairwise distances are known, an intuitive approach is to define the neighbors of a point, xi, as those within a certain distance threshold, or, equivalently, inside an n-dimensional ball around xi. A kernel function assumes the role of this ball, by assigning values to neighboring points as a function (discrete or continuous) of how close they are to xi. The question becomes: what kernel size should be used for each point? 

## 2.1 Similarity kernels 

Consider a set of points X ∈ Rn. Typically, a symmetric, positive semi-definite similarity ker-nel [93] is chosen to determine weighted connections between data points based on the ambient Euclidean distances between them. For each pair of data points xi, xj ∈ Rn, it returns a number between 0 and 1 which determines how close, or strongly connected, they are. This effectively defines a neighborhood around each point. 

2.1.1 Discrete kernels 

Possibly the simplest choice for a kernel is the ε-neighborhood [e.g., 10]: 

Kij (ε) = 

{1, if ‖xi − xj ‖ < ε 

0, otherwise , (1) where ‖ · ‖ is typically the Euclidean norm in Rn. This results in discrete-like neighborhoods whose sizes may be quite sensitive to the choice of ε, so implicit is the assumption that sampling is approximately uniform. Instead of defining a neighborhood radius, a more common approach is to specify the number of neighboring points, k. Letting Nk(xi) be the set containing the k points closest to xi in Rn (not 5including xi1), a k-nearest neighbors kernel can be defined as: 

Kij (k) = 

{1, if xj ∈ N k(xi)0, otherwise , (2) which is commonly symmetrized by making Kij (k) = 1 if xj ∈ N k(xi) ∨ xi ∈ N k(xj ).

2.1.2 Continuous kernels – global scale 

In order to have the kernel values decrease with increasing distance between data points, a Gaussian kernel is commonly used: 

Kij (σ) = exp 

(

−‖ xi − xj ‖2

σ2

)

. (3) This gives a continuous similarity scale from 1 (when xi and xj are identical) down to some predetermined cutoff below which the kernel is considered to be zero (meaning no connection in the data graph). Such a threshold is typically chosen to be a very small value, often at the limit of numerical precision, and is often required to ensure compactness of the kernel. One would like the parameter σ to be just large enough to be able to capture local manifold patches. There are several heuristics for finding such a scale: the median of all pairwise distances in X (or another percentile), the mean (or median) of the distances to each point’s kth nearest neighbor [66], or a scalar multiple of the maximal distance from a point to its nearest neighbor in the data [59]. Also common is to choose a scale so that each data point is sufficiently connected to at least one other point [67]. A different approach is based on inspection of the curve given by the sum of pairwise kernel values. When the double-sum ∑ 

> i,j

Kij (σ) is plotted against σ using a log-log scale, the slope 

d log ∑ 

> i,j

Kij (σ)d log σ (4) is proportional to the intrinsic dimensionality of the data [30]. A global scale is then chosen from within a linear region of such curve. In [52], a similar procedure is proposed that considers, instead, the curve given by the weighted average of the degrees Zi(σ) = ∑ 

> j

Kij of each data point xi, after taking the logarithm: 

〈log Zi(σ)〉 =

∑ 

> i

log Zi(σ) · (1 /Z i(σ)) 

∑

> i

(1 /Z i(σ)) , (5) The use of the inverse of each point’s degree as weights is intended to compensate for density heterogeneities. The choice of σ is then made precise by choosing the argmax of the slope of 

〈log Zi(σ)〉 plotted against log σ, which in many cases should occur near the center of the lin-ear region of equation 4. One complication occurring in both approaches, however, is that more than one linear section (and, equivalently, more than one local maximum of the slope) may exist, requiring that additional criteria be defined to make the choice of σ truly automated.  

> 1Throughout, when referring to a point’s set of k-nearest neighbors, we shall not include the point itself (unless otherwise stated), and further assume that no two points are identical.

62.1.3 Continuous kernels – multiscale 

A more localized strategy is to use a multiscale kernel , where each point has an individual scale, or bandwidth. Instead of a single, global scale, there are now N parameters. The advantage is that, if the scale selection is adequate, the kernel may capture the characteristics of more complex datasets and manifolds that have non-uniform sampling and geometry. In the self-tuning method [111], local scales are used in a Gaussian kernel by replacing the global scale σ, from equation 3, by √σiσj , where σi and σj are the scales assigned to xi and xj ,respectively. This results in the symmetric kernel: 

Kij (σi, σ j ) = exp 

(

−‖ xi − xj ‖2

σiσj

)

. (6) Each σi is set as the distance to the kth nearest neighbor of xi; authors recommend k = 7 [111, 79]. In [17] and [16], a variable bandwidth kernel is proposed that combines the use of local band-widths with a global scale parameter, . The kernel then takes the form: 

Kε (xi, xj ) = exp 

(

−‖ xi − xj ‖2

4(q(xi)q(xj )) β

)

, (7) where q is a local density function and β an additional (non-positive) parameter. An initial estimate for the local bandwidth around each point xi is set as the square-root of the mean squared distance to the k-nearest neighbors of xi, with k = 8. Finally,  is automatically tuned as the argmax of equation 4 above; however, the authors do not consider cases in which more than one local maximum may exist. Other methods also adopt individual bandwidth parameters, but use asymmetric kernels that are symmetrized a posteriori . In the t-SNE algorithm [101], the single-scale Gaussian kernel 

Kij (σi) = exp 

(

−‖ xi − xj ‖2

2σ2

> i

)

(8) gives a measure of affinity, or similarity, between pairs of points. It is then normalized as 

pj|i(σi) = Kij (σi)

∑  

> k6=i

Kik (σi) (9) to yield transition probabilities, and finally symmetrized as 

pij (σi, σ j ) = 12N

(pj|i(σi) + pi|j (σj )) . (10) Each σi is fit to xi so that the distribution of pj|i, ∀j attains entropy Hi such that its perplexity, 

2Hi (a real-valued number representing the “effective number of neighbors”), approximates some prespecified value, k. The authors recommend a value for k between 5 and 50. In the UMAP algorithm [77], an exponential kernel is used instead of the typical Gaussian. Using a prespecified neighborhood size, k, let Nk(i) be the set of k-nearest neighbors of xi. With 

ρi as the distance to the nearest neighbor of xi, the kernel has the form 

Kij (σi) = exp 

(−max {0, ‖xi − xj ‖ − ρi}

σi

)

, j ∈ N k(i), (11) 7and is symmetrized as 

Uij (σi, σ j ) = Kij (σi) + Kji (σj ) − Kij (σi)Kji (σj ). (12) It can be seen as a hybrid between continuous and discrete, since Uil is set to zero for any point xl

not in Nk(i). Each σi is fit to xi so that ∑ 

> j

Kij (σi) approximates log 2 k (loosely analogous to the perplexity approach from t-SNE). 

2.1.4 Adaptive neighborhood size methods 

Other methods attempt to automatically determine optimal neighborhoods. Most of these are based on determining an optimal k for a k-nearest neighbors ( k-NN) graph; this can be done either globally or by selecting a local neighborhood size ki around each point xi, known as adaptive neighborhood selection [102]. Some approaches optimize a global k based on its performance in a specific embedding algo-rithm. For instance, the method from [91] is tailored to Isomap [96], while others [65, 3] apply to LLE [88]. In [3], a local method is additionally proposed that produces a nearest-neighbor graph with variable ki, under the assumption that the manifold is connected. Others are based on first estimating the local tangent space around each point, then setting ki to include as neighbors those points that are close to it. Such methods [e.g., 106, 78] typically work with positional information for the tangent space computation (usually via SVD). Also available are methods that are not based on the nearest-neighbors concept. In computa-tional geometry, the idea of refining an initial estimate of connectivity from a simplicial mesh has been used before, usually specific to the case when d = 2 and n = 3, i.e., surfaces in 3-D space [5, 4, 14, 12]. Other approaches extend this idea to arbitrary dimension [13, 19], but still require knowledge of d. Most of the algorithms in this class use point clouds as input, so they can exploit positional information to decide on the appropriate neighborhood/connectivity. Among the myriad ways of estimating neighborhoods, there is little agreement on which is most successful; see [71] for a review. Before proceeding to our algorithm, then, it is helpful to first understand what makes this such a hard problem. How can it fail, and what requirements must it fulfill in order to properly capture the topology and geometry of M? This brings us to the geometry of manifolds. 

## 2.2 Reach and the geometry of manifolds 

The neighborhoods implied by a kernel should agree with M, or at least approximate a tubu-lar neighborhood of it. As exemplified in Figure 1, if neighborhoods are too small, the implied manifold may become disconnected, i.e., falsely divided into disjoint sub-manifolds or clusters [91]; if too large, they may cause M to self-intersect, collapsing bottlenecks or curved regions, or cause “smoothing,” or “folding.” Such shortcomings are well-known in the manifold infer-ence literature—while the former case typically occurs due to non-uniform sampling, the latter is mainly caused by an incompatibility between the sampling rate and the reach of M [41, 97]. We now expand on these points. Letting the medial axis of M be the set of points in Rn with at least two closest points in 

M, the reach, τ , can be defined as the minimum distance from M to its medial axis. Locally, it is constrained by the minimal radius of curvature (i.e., maximal curvature of a geodesic through 8M); globally, it is constrained by the presence of bottlenecks (Figure 3). The reach encodes essential geometric properties of M, and has been widely used in the manifold learning community [4, 12, 82, 19, 83, 49, 73, 43, 1, 20]. It approximates the size of the largest ball in ambient Rn such that points in M can be seen as lying in Euclidean space Rd [18]. A related concept, the local feature size of a point xi ∈ M , is the smallest distance between xi and the medial axis of M, so 

τ can be seen as the infimum of the local feature size anywhere on M [13]. 

Figure 3: The reach, τ , is a measure of the shape of a manifold. ( A) A 1-dimensional manifold M

with a bottleneck; the reach (double arrow) is the smallest distance between M and its medial axis (dashed curves). ( B) A highly curved manifold; now the reach indicates the high curvature region. When τ is positive, it provides a measure of the “local distortion” [18]; the larger it is, the easier inference becomes. Some authors [e.g., 81, 42] assume large reach in order to test the manifold hypothesis and to find bounds on the required sample size. In [18], the reach is used when establishing bounds on the quality of an intrinsic dimensionality estimation based on k-nearest neighbors. Obtaining a good representation of M, therefore, requires consideration of its reach. In terms of our problem of finding an appropriate kernel, this effectively means that no neighborhood radius should cross the medial axis of M.Sampling is a further complication, and essentially what makes this a hard problem: when it is nonuniform and sparse (common in real-life datasets), it is not always clear whether the space between points constitutes an undersampled piece of M, a hole, or a gap between disjoint subman-ifolds (cf. Figure 2). The latter two conditions, of course, relate to reach. Narayanan and Mitter [81] prove that the number of required samples depends polynomially on curvature, exponentially on intrinsic dimension, and linearly on intrinsic volume. Aspects of our algorithm address each of these during the iteration process. In all such cases, choosing a globally-fixed radius is likely to be problematic. While defining neighborhood size based on a fixed number k of neighbors can be helpful to deal with nonuniform density (since the neighborhood radius adapts to the local pairwise distances), it is bound to violate the reach if k is too large. It will also be a problem when the intrinsic dimensionality is not constant throughout M, as higher dimensions require exponentially more neighbors. Mekuz and Tsotsos [78] point out the lack of a principled way for setting this parameter, which in practice is often tuned empirically based on prior knowledge of the desired output. As put by Wang et al. [106], the effectiveness of manifold learning algorithms depends on how nearby 9neighborhoods overlap and on the interplay between the curvature of the manifold and sampling density. In terms relevant to this paper, the neighborhood radius should be smaller than the local feature size, but large enough to account for sampling variability and local dimensionality. We propose an iterative approach to developing a kernel, so that it can adapt appropriately to the neighborhood characteristics around each point. 

# 3 The algorithm 

We here overview our algorithm for finding the neighborhood scale around each point in a manner that makes it globally consistent as a covering of the data points. As is common in manifold learning, we start with a pairwise distance matrix, not the points themselves. The first step is to build a graph in which each datum is connected to an appropriate neighborhood containing other data points. This data graph defines a topology; we refer to it as the neighborhood graph .As we reviewed above, in the discrete case one might choose k-nearest neighbors, while in the continuous kernel case there is a bandwidth parameter that effectively defines a “ball of influ-ence” around each point. Scale is the radius of such a ball; a level set of the kernel function that essentially contains those neighbors whose weights are non-trivial. Our goal, then, is to find those scales—or neighborhoods—that support non-linear dimensionality reduction, geodesic estimation and, in general, manifold inference from the given pairwise distances. We do not have sampling guarantees, so will develop a statistic to check whether reach and curvature constraints might be violated. 

## 3.1 Subtleties of scale 

Since scale may not be constant across the data set, we argue that it should be the first property to be inferred from the data. We start by imposing the manifold assumption, but from an empirical perspective. Unlike most theoretical studies, we do not assume the manifold is pure, i.e., that it has constant dimension. In a simple case, the data may be drawn from a union of different mani-folds whose dimensions are not known a priori —such datasets have been considered infrequently, although exceptions exist [e.g., 53, 73]. Second, we do not know the sampling rate, or density. Rather, we build it up, conservatively, with putative nearest neighbors to each data point, by imposing a necessary (but not sufficient) condition. These putative neighbors will be refined, as the algorithm iterates, to achieve sufficiency. While the manifold assumption does imply the existence of local neighborhoods, their size may vary over the dataset; we require that the sampling be nearly constant over each of them. In effect, the density of points must be determined locally while respecting the global manifold geometry. We illustrate the complexity of this situation in Figure 4. Shown is a data sphere with an apparent spike emerging from it. On one hand, such complex datasets could derive from two unrelated systems, which only appear to connect through their embeddings. On the other hand, the data could derive from a non-linear system that includes two regimes, one responsible for the spherical data and the other for the spike. To handle the first situation, we must allow datasets to consist of unions of manifolds. This suggests the interpretation in Figure 4-B, where the separation is obscured by sampling. Since manifolds with boundary and high curvature are also possible, the 10 situation in Figure 4-C arises. There is an apparent change in intrinsic dimension due to the small reach in the spike and the large boundary curvature. Because the (3-D) spike is so narrow, sampling suggests it is 1-dimensional, while the bulk of the points derive from a 3-D manifold. We submit that such situations occur in real datasets and, since the data are fixed, we cannot appeal to knowing the sampling density or the manifold dimensions and reach. Instead, we address the interplay between manifold reach and sampling density pragmatically. Along the spike, the data appear to be 1-D; in the ball, 3-D. We seek a neighborhood graph that supports these inferences, so “most” points enjoy a neighborhood that agrees with their apparent dimension. At the join (or high-curvature neck), it is unclear. Moving from the spike to the ball suggests that dimension should be increasing; from the ball to the spike, it should be decreasing. For the neighborhood graph, most points along the spike should see ∼2 neighbors, and most points in the ball should see ∼23 neighbors; the problematic points should see something intermediate. Such results will be shown to follow from our algorithm. We claim that either of the alternatives is worse; one should not impose an apparent dimension-ality (or connectivity in the neighborhood graph) globally. To wit, if small numbers of neighbors (appropriate for the spike) are enforced on the ball, then holes are likely to be introduced. Or, if too many neighbors are enforced on the spike, it will collapse on itself. Both change the topology drastically (these situations are illustrated later, in Figures 24–25). Figure4 subtleties-new 

Figure 4: The “manifold” subtleties of complex datasets. (A) Sampled data from a non-linear system that includes two regimes. (B) It may be the case that the data in each regime define separate manifolds, shown by color. After sampling their union, however, the evidence for the separation is absent. (C) Or the data may be drawn from a single, connected manifold whose geometric properties change rapidly. In both cases, the intrinsic dimensionality appears different in the spike vs. the ball. Colored meshes indicate underlying manifolds. 

## 3.2 Overview of the algorithm 

Let the dataset, X , be a sampling of a (possibly non-pure) manifold M = ∪αMα, with the dimension of each component Mα denoted by dα. It consists of N points in ambient space Rn,where n ≥ dα, ∀α. The manifold may have a boundary, and the number of components is not known a priori .We work with two graphs: the first unweighted, and the second with edge weights given by a kernel. Our strategy is to begin with a conservative estimate of the unweighted graph, and extend it to a global weighted graph that suggests an estimated manifold covering. The validity of this 11 extension is evaluated by a measure of volume in both graphs; an iterative algorithm is used to infer individual local scales for each point xi. Before presenting the algorithm, we introduce the two graphs. Let the unweighted graph be G = ( V, E ), with |V | = N and adjacency matrix A with entries 

aij , where to each point xi ∈ X is associated a node i ∈ V . We denote its initial estimate by G(0) ;successive refinements are indicated as G(t) until convergence ( G?). Since we seek a scale for each data point, we work with a multiscale Gaussian similarity kernel, defined as in section 2.1: 

Kij = exp 

(−‖ xi − xj ‖2

σiσj

)

. (13) The kernel value Kij is therefore symmetric and equivalent to that of a traditional Gaussian kernel (equation 3), except using the geometric mean of σi and σj as its scale. Notice, in particular, how the scales and the kernel value are coupled: setting the scale incorrectly could make distant points 

xi and xj appear close in similarity. Given a set of individual point scales σi (sometimes collected into the vector σ ∈ RN ), we define a second, weighted graph G = ( V, E, W ) as the complete graph on all pairs of data points in X . Its weighted adjacency matrix, W , has entries wij = Kij .While the unweighted graph will be related to nearest neighbors and computational geometry, the weighted graph will be related to spectral methods on manifold inference. In particular, we expect the Laplacian of G to approximate the Laplace-Beltrami operator on M, subject to the number of data points and their sampling. The algorithm is initialized by computing a coarse estimate of G. As described later in sec-tion 3.3, this is achieved by exploiting the geometry of medial balls between pairs of points to produce a Gabriel graph [48, 76]. A Gabriel graph is that in which there is an edge between two points xi and xj if and only if they are the only two closest points to the midpoint of the line segment joining them. The main advantages of using a Gabriel graph as a starting point are: ( i)it is scale invariant, so a prespecified ε-neighborhood (equation 1) is not required; ( ii ) there is no global constant k (it can vary); and ( iii ) neighbors are not limited to the closest neighbors in ambi-ent space. Thus, it allows for connections to “jump across” sampling gaps while keeping the data graph sparse. However, as described in section 2.2, obtaining a good inference of M amounts to finding reasonable estimates of its reach and local feature size. For that to occur, no edge segment `ij 

between two points xi and xj should cross a medial axis of M. As the examples that follow will show, there are several cases in which the Gabriel graph will violate this. Therefore, additional steps are necessary to refine it. The Gabriel graph provides a necessary condition (all the correct connections are present, but possibly others as well); our refinement moves toward sufficiency. In order to estimate G—the weighted counterpart of G—we will use the weights that are ob-tained by applying a continuous kernel (equation 13) over the points in X . Such a kernel requires scales, or bandwidths, σ that must be estimated from G. These will be obtained from an optimiza-tion procedure that finds the smallest such scales ensuring that all discrete edges have a minimum kernel value as weight. At this point, a weighted graph G can be obtained from σ.It is now helpful to articulate the geometry more carefully; Figure 5 depicts how the discrete connectivity relates to the manifold geometry. In particular, for a real dataset, the few closest points surrounding xi are the best candidates for “nearest” neighbors—this is all that can be asserted locally. Let pi and pj be the projections of two neighbors xi and xj onto M, respectively. Then, 12 any point along the geodesic between pi and pj should be closer to no sampled point other than 

xi or xj . By further assuming xi ∈ M , ∀i or at least that ‖xi − M‖ Rn < ε, ∀i and small ε,then ‖xi − xj ‖Rn approximates the geodesic when the curvature between pi and pj is small. Equivalently, the line segment `ij between xi and xj lies on the tangent space TpM, where p is the midpoint between pi and pj ; see Figure 5. The existence of a geodesic follows from identifying the tangent plane that includes the points with the exponential map of the manifold around them. Such an “edge-centric” approach connects differential geometry to the underlying graph. This is illustrated in Figure 5, where the kernel values are shown as shading in the tangent plane. Notice how xi and its neighbor xj both fall under the bright kernel values; i.e., they are very similar (in this measure) to each other. Stated in geometric terms, we assume that the neighbors lie within the injectivity radius around p. In fact, we will show (Figure 14) that the value of a multiscale kernel between two data points is equivalent to that of a rescaled, single-scale kernel centered at the midpoint between those two points. 

Figure 5: Relating the discrete neighborhood graph to manifold geometry. Nearby sampled points (i and j) on a patch of manifold M lie in (or near) the tangent plane TpM to the midpoint ( p). Line segments (edges) between neighboring points lift, via the exponential map, to geodesics in 

M. The continuous kernel extends this discrete relationship to the full tangent plane. The values of the kernel centered at p are shown as shading, extending in every direction in TpM. Our algorithm shall enforce this relationship, i.e., the consistency between discrete edges and large kernel values. Now, the optimized scales can be used to evaluate the current approximation and identify the edges in G that are “too expensive,” i.e., are likely to violate the local feature size. We proceed by computing successive refinements of both G and σ, in an iterative manner, until no further change is observed. We then return the final version of the discrete and weighted graphs (denoted by G?

and G?, respectively). 13 One can view the computation of G as a relaxation of the discrete connectivity in G. In fact, as we shall see in section 3.5, a relaxation statistic, δ′

> i

, will be used to prune discrete edges that produce a poor approximation. More specifically, when a node i with degree, deg( i), in G has δ′

> i

close to 1, it means i has retained approximately the same degree in G, only continuously spread as a Gaussian around it. Each of the steps above are listed in Algorithm 1 and will be described in detail. We begin with the discrete connectivity rule (Gabriel graph); then the scale optimization is developed, followed by the edge-pruning step. Figure 6 illustrates the results of our algorithm on datasets for which the Gabriel graph alone cannot infer a good approximation of the manifold connectivity. 

Algorithm 1 Iterated Adaptive Neighborhoods kernel  

> 1:

procedure IAN KERNEL (D) . Input: distance matrix, D 

> 2:

G(0) ← GABRIEL GRAPH (D) . Compute initial G (sec. 3.3)  

> 3:

repeat Iteration  

> 4:

σ(t), ← OPTIMIZE SCALES (G(t), D ) . Update scales σ (sec. 3.4)  

> 5:

G(t) ← MULTI SCALE KERNEL (D, σ(t)) . Weighted graph (eq. 13)  

> 6:

δ′, C ← COMPUTE VOLUME RATIOS (G(t), σ(t)) . Statistic δ′ (sec. 3.5.1)  

> 7:

G(t+1) ← SPARSIFY (G(t), δ ′) . Update G (sec. 3.5.2)  

> 8:

until no further change in G 

> 9:

return G?, G?, σ? . Output: final graphs and optimal scales  

> 10:

end procedure 

## 3.3 Neighbors in a Gabriel graph 

We begin by defining a set of putative neighboring points of xi (denoted by N (i)), which uses the connectivity rule found in a Gabriel graph [48, 76]. It directly incorporates the observation that closest neighbors should have no points “between” them. 

Remark 1. Two points, xi and xj , are Gabriel-nearest neighbors to each other if and only if they both touch the same closed ball, Bij , that is empty except for xi and xj .Note that Bij is therefore a medial ball , i.e., a ball whose center point is a medial axis (with respect to the set of sampled points). Thus, this connectivity criterion can be restated as creating an edge for all those medial balls, and only those, touching exclusively two points (to be clear, if a third point touches Bij no edge shall be formed between xi and xj ). Hence, to each edge eij is associated a medial ball Bij centered and the midpoint between xi and xj with radius ‖xi − xj ‖/2

(see Figure 7). This is furthermore equivalent to the following alternative definitions: 

Remark 2. Points xi and xj are Gabriel-nearest neighbors if and only if any point along the line segment `ij = xixj in Rn has either xi or xj (or both) as its only closest point(s). 

Remark 3. In terms of the Voronoi diagram [44] of X (with the cell around xi denoted by Vi), xi

and xj are neighbors when `ij crosses a single Voronoi hyperplane Hij (namely that between the cells Vi and Vj ) and the midpoint between xi and xj is in Hij .14 G (0) 

A G ⋆

¾⋆ G⋆

G (0) 

B G ⋆

¾⋆ G⋆Figure 6: See next page. 

15 Figure 6: Steps of Algorithm 1 on toy datasets. ( A) Dataset with several challenges: non-uniform density, non-uniform dimension, and high curvature. After pruning 6 edges (dashed red lines) from the original Gabriel graph, G(0) , the algorithm converges, inferring reasonable discrete neighbor-hoods ( G?); the optimal scales σ? produce a weighted graph G? whose connectivity closely approx-imates that of G?. ( B) Dataset with three Gaussian clusters of non-uniform density. The Gabriel graph approximation, G(0) , naively connects all clusters using multiple edges. After convergence, the clusters become disconnected in G?, and its weighted version follows this by assigning negli-gible weights (due to σ?) between points in different clusters. As a concrete example (refer to Figure 7), consider two points xi and xj at a distance rij 

from each other, with midpoint p. Assume the region in the manifold between them is uniformly sampled. Now consider the ball centered at p with radius rij /2, therefore touching xi and xj . If there are no points in its interior, we say xi and xj are nearest neighbors. Conversely, if it contains other points in its interior, under our assumption of uniform density this means that there is at least one other point xk “between” xi and xj . So we say that xi and xj are not nearest neighbors, in the sense that connecting xi and xj directly would be “crossing over” xk; this implies that an edge eij in the resulting graph would be a poor approximation to a geodesic in M (i.e., if M is “locally uniformly sampled,” the segment `ij would be passing outside of M). Note that, even when the input to the algorithm is solely a distance matrix (i.e., with no position information), this connectivity criterion may still be evaluated by considering the triangle xi–xj –xk and using Apollonius’s theorem to compute the length of the median from xk to p (Figure 7-D). 

Figure 7: Connecting “nearest neighbors.” ( A) A set of data points in space. ( B) An edge can be formed between xi and xj because there is no other point in the interior of the ball Bij centered halfway between xi and xj . ( C) Here, because of the presence of a third point xj inside Bij , xi and 

xj cannot be neighbors. ( D) Even in the absence of the original data point coordinates, i.e., given only the distances between all pairs of points, Apollonius’s formula can be used to determine the length of the segment p–xk, where p is the center of Bij . Namely, p–xk is a median of the depicted triangle. Here, because the length of the median is less than the radius of Bij , xi and xj cannot be neighbors. ( E) Edges are drawn connecting points xi to xk and xk to xj because both Bik and Bjk 

are empty except for those pairs of points, respectively. The Gabriel graph is a subgraph of the Delaunay graph [37], and enjoys a number of key properties [76]. We emphasize: ( i) they are scale invariant, i.e., there is no pre-specified threshold on the diameter of medial balls that can form connections; ( ii ) the guarantee that Gabriel graphs 16 connect points to their true nearest neighbors when M is uniformly sampled as a grid (shown in Figure 9); and ( iii ), Gabriel graphs provide a locally-adapted neighborhood size ki, for each point 

xi, based on the local geometry. Crucially, they do not require an initial guess of the number of neighbors, of the intrinsic dimensionality, or of a maximum neighborhood radius. Nevertheless, the neighborhoods given by the Gabriel graph are not sufficent. We now expand on a few of their properties—these will be useful in motivating the rest of the algorithm. 

3.3.1 Closing triangles 

Here we show that the edges created using the above connectivity rule can only form acute triangles in Rn. Let three points xi, xj , xk be such that xi and xk are connected, as well as xj and xk. The rule says, xi and xj shall be connected only if xk is outside the closed ball Bij of radius R = rij /2

centered half-way between xi and xj (where rij stands for the Euclidean distance between xi

and xj ). Using Apollonius’s formula for the squared distance m2 between xk and the midpoint between xi and xj , we obtain 

m2 = 14(2 r2 

> ik

+ 2 r2 

> jk

− r2 

> ij

). (14) Then, xk is in Bij if and only if m2 ≤ R2, so 

14(2 r2 

> ik

+ 2 r2 

> jk

− r2 

> ij

) ≤ R2 = ( rij 

2 )2

r2 

> ik

+ r2 

> jk

≤ r2 

> ij

.

(15) Notice that equality will hold when xi–xj –xk is a right triangle. Therefore: 

Remark 4. A triangle will be formed by edges in a Gabriel graph only when it is acute (see Fig-ure 8-A). 

3.3.2 Maximum curvature 

The above result leads to a bound on the maximum principal curvature that is allowed locally on 

M such that the Gabriel graph correctly approximates it (i.e., without closing a triangle). Assume 

xi, xj , and xk are points in a smooth manifold M as in Figure 8-B, up to the level that the sampling defines. If we assume that the curvature, κ, is locally constant, then the geodesic from xi

to xj passing through xk is an arc of a circle C. Therefore, the segments `ik and `kj approximate geodesics on M, but not `ij (which would cause “folding”). Hence, values of curvature that can be correctly inferred are those that do not create an edge between xi and xj (i.e., those for which the ball Bij is non-empty). In this case, from equation 15, the maximum such curvature, κmax , occurs when xi, xk, and xj form a right triangle in space (as any larger value would cause this triangle to be acute, connecting xi to xj ). Then, from Thales’s theorem, the diameter D of C would equal that of the hypotenuse `ij , so 

κmax = 1

D/ 2 = 2

D = 2

√

r2 

> ik

+ r2

> jk

. (16) 17 Figure 8: Implications of the connectivity rule in a Gabriel graph. (A) Closing triangles from edges: three points will be mutual neighbors if and only if they form an acute triangle (left). If the angle between xi and xj at xk is at least π/ 2, all three points will lie in Bij , so no edge is cre-ated (right). ( B) The maximum principal curvature in M (shown in blue) that can be reasonably approximated by the resulting graph geodesic (path) is constrained by the sampling interval. The limiting case occurs when three points form a right triangle (top), cf. equation 16). When sam-pling is too sparse (bottom left), a triangle may be formed, in this case preventing the graph from adequately capturing the manifold’s geometry. As sampling frequency increases (bottom right), higher curvatures can be better approximated. A special case to consider is when M is uniformly sampled with constant interval T over arc length. Then, the arc length s between i and j is 2T ; but, since rij = D, s covers half the circle and we have 2T = πD/ 2. Equation 16 then becomes 

κmax (T ) = π

2T . (17) 

Remark 5. Equations 16 and 17 define the maximum geodesic curvature in M that can be ade-quately inferred from a Gabriel graph. As a consequence, the reach is lower-bounded by 1/κ max .

3.3.3 Degree distribution in Gabriel graphs 

We now study the above connectivity rule starting with flat, uniformly sampled manifolds (i.e., “regular grids”) to illustrate how Gabriel graphs naturally adapt to both their geometry and dimen-sionality. As shown in Figure 9-A, in such ideal cases the degree of an interior node in the Gabriel graph agrees with the true number of (literal) nearest neighbors, i.e.: 2 for collinear points, 4 for a square grid, and 6 for a triangular grid. Node degree appears to grow with dimension as 2d, except for the triangular grid (which, in some sense, looks too “non-generic”). Adding noise (Gaussian, with standard deviation equal to half the spacing between neighboring points) supports this conjecture, as the degree then ap-proaches 2d regardless of the original grid structure. This holds in higher dimensions as well, for both normal and uniform sampling at random (Figures 9-B,C and 12). 18 Remark 6. The expected number of neighbors in a Gabriel graph approximately follows a dis-tribution centered at 2d (where d is the intrinsic dimension of the data) for a variety of sampling strategies (Figure 9-C). Importantly, because Gabriel graphs are inherently scale invariant, this degree distribution is largely independent of sampling density. How to explain such remarkable regularity despite the randomness of sampling? A comple-mentary geometric view of the Gabriel graph connectivity rule is illuminating: each edge between data points implies an “occluding hyperplane” that blocks other points from becoming neighbors (see Figure 10). For example, when d = 1, two points necessarily occlude any additional connec-tions, and every non-boundary point must have 2 neighbors. Now, using the diagrams in Figure 11 as reference, we find that, when d = 2, on average ∼4 points are sufficient to occlude a point xi

from all sides. For d = 3 this number is doubled again, and the expected number of neighbors becomes ∼8, revealing the trend. Every additional dimension adds a new coordinate axis along which the previous constraints are duplicated, roughly doubling the average number of directions available from which neighbors can connect. Once 2d balls are “attached” to xi, the remaining space is greatly reduced, and so is the probability of drawing a sample point from inside the region 

H enclosed by the hyperplanes. When the neighbors are regularly spread around xi, by construction this region H is equivalent to a d-dimensional orthoplex 2 (or cross-polytope). A d-orthoplex has 2d facets (or (d-1) -faces), and is one of the three finite, regular, convex polytopes that exist in dimension higher than 4 (the other two being hypercubes and simplices). Naturally, when sampling is not uniform, we should find irregular orthoplexes instead. While this geometric construction supports our empirical results, and implies they should hold in higher dimensions, it also suggests the following: 

Remark 7. Our experiments on the growth in dimension of randomly sampled points agree with a model in which Gabriel neighbors lie approximately in the facets of an orthoplex. We shall later use the additional observation that the dual polytope (a d-hypercube) of an orthoplex is obtained by placing a vertex (i.e., a neighbor) in each of its 2d facets. The Gabriel graph enjoys many attractive properties, and provides the starting point for our algorithm. The above arguments show how the space is largely filled by “Gabriel balls” within the manifold, but such balls may also fill space across holes and bottlenecks; curvature must be dealt with. Examples were given in Figure 6, where we showed that Gabriel connections can arise incorrectly and must be removed. To do so, one must “look” in every direction (of the tangent plane), and past immediate neighbors. For this, we now develop the weighted graph counterpart to the Gabriel graph, exploiting the kernel to extend local information globally. This begins to connect the graph construction more directly to manifold properties. 

> 2An orthoplex is a line segment in 1-D, a square in 2-D, a regular octahedron in 3-D, a 16-cell in 4-D, etc.

19 Figure 9: Regularity of node degree distribution in Gabriel graphs with random sampling. (A)Node degree in graphs computed from regular grids (constant sampling interval, T ) and their jit-tered versions (Gaussian noise with std. dev. 0.5T ). Top: A sequence of collinear points (left) produces a one-dimensional grid (center). Addition of noise (right) does not change the mean de-gree (constant 2 for interior points). Middle: A square grid (left) results in a quadrilateral mesh with constant degree 4 in its interior. Although addition of noise considerably scrambles the points, the mean degree is roughly unchanged. Bottom: Points arranged as a triangular grid (left) result in a triangular mesh where every interior node has degree 6. Its noisy version looks similar to a noisy square grid, with mean degree also approximating 4. (Cont. next page.) 

20 Figure 9: (Cont. from previous page.) (B) Degree distribution for interior points of d-dimensional triangular and square grids after addition of Gaussian noise. Moderate amounts of noise are suffi-cient to make the mean degree become approximately 2d. Error bars indicate standard deviation; dotted lines show constant 2n values for reference. ( C) Mean degree of d-dimensional manifolds sampled using different strategies: uniformly at random, normally at random, and as jittered ver-sions of regular triangular and square grids (as in ( A), added Gaussian noise with std. dev. 0.5T ). Remarkably, mean degree grows approximately as 2n regardless of the sampling strategy. 

Figure 10: (A) A central point xi (in blue) and its neighbors (in black). Every neighbor xj of xi

will “occlude” the entire area behind a hyperplane tangent to Bij at xj (dashed lines). That is, no point inside the occluded areas (shaded region) can form a connection with xi. Here, the dashed ball does not form a connection between xi and xk because xj lies exactly on its boundary; despite this, xk still contributes with an occluding hyperplane, preventing farther points from connecting to xi. ( B) In principle, there is no limit to the number of neighbors a point in ambient space Rn may have (when n ≥ 2); e.g., any number of points lying exactly on a hypersphere around xi (dotted curve, in orange) will not occlude one another. Sets of nodes with connectivity such as this are termed “wheels” in graph theory, and the more points they contain, the less likely they are to occur in real datasets. In this example, any appreciable variability in the distance from xi to its neighbors would cause one (or several) of them to become occluded. ( C) Points inside occluded areas can also contribute with additional occluding hyperplanes. Here, although xk lies inside the region occluded by xj (and therefore cannot form a connection with xi), it produces further occlusion behind a hyperplane of its own (region shaded in red). So xl cannot connect to xi, either, due to the presence of xk (even though it is not occluded by xj ). 21 orthoplex Figure 11: Occlusion hyperplanes (shown in gray) due to neighbors in dimensions 1, 2, and 3 (A–C, respectively); compare with Figure 10. Every additional dimension adds a new coordinate axis along which the previous constraints are duplicated, roughly doubling the average amount of directions available from which neighbors can connect. Once 2d Gabriel balls are “attached” to xi,the remaining space is greatly reduced, and so is the probability of drawing a sample point from inside the region enclosed by the hyperplanes. 22 Figure 12: Distribution of node degree in the Gabriel graph of datasets with different sampling strategies and dimensionalities. Top: Points sampled normally (blue) or uniformly (orange) at random from a two-dimensional ball result in similar degree distributions centered at 22. Bottom: 

In higher dimensions, interior points continue to follow this pattern. On the left, a 4-dimensional unit ball sampled uniformly at random is shown projected onto R3, with boundary points labeled as those with vector norm > 0.9 (edges omitted for clarity). It produces a Gabriel graph where interior points have degree distribution centered at ∼24, and the mean degree of boundary points is close to 23.23 3.4 Multiscale optimization 

We now begin to develop the iteration in Algorithm 1, given the initial Gabriel neighborhood graph, G(0) . Assuming (temporarily) that this gives correct local neighborhoods, what should the corresponding scales be for a Gaussian kernel? In effect this is an extension of G into a weighted counterpart, G. From Figure 5, this weighted graph is also a type of approximation of (aspects of) the continuous manifold. Because density is not necessarily uniform, different points might have different neighborhood radii, so a multiscale Gaussian similarity kernel (equation 13) is used. Each point xi has its own associated scale, σi. To develop the computation of such scales, we now move into the continuous domain and exploit the geometric notion of a cover. 

3.4.1 Covering criterion 

A criterion for separability between two Gaussians has been developed in the mixture-of-Gaussians literature [33, 103, 6]: two spherical Gaussians, i and j, can be distinguished (in the sense of solving a classification problem) with reasonable probability when they have a separation of at least 

‖μi − μj ‖ > C max {σi, σ j }, (18) at which the overlap in their probability mass is a constant fraction [103]. We flip this around by using a different, but related, construction: consider Gaussians now centered at the midpoints (i.e., not on data points) to indicate whether nearby points should be connected, not separated (Figure 5 illustrates this construction directly). Furthermore, because we use a multiscale kernel (equation 13), the (non-normalized) Gaussian density becomes a function of √σiσj . Hence, we obtain a criterion for what we term C-connectivity :

Definition: Two neighbors i and j in the discrete graph G = ( E, V ) are C-connected by the multiscale kernel when the geometric mean of their individual scales is at least the distance between xi and xj scaled by a positive constant, C:

C‖xi − xj ‖ ≤ √σiσj . (19) The constant C plays a role in normalizing for unknown density; it will be developed in sec-tion 3.5.2. For now, we illustrate its role in the connection from graphs to manifolds. Figure 13 shows the graph over a set of data points, and the local scales obtained (by the algorithm below) for different values of C. Choosing C too large yields scales (and therefore Gaussians) that are too large, that is, their overlap has peaks. Choosing it too small yields scales that introduce holes. Choosing it correctly, the Gaussians form a covering of the manifold that approximates a partition of unity. Such partitions of unity are used in differential geometry to extend local information (in our case, the scales) to global information (a covering of the manifold). By choosing appropriate scales, i.e., scales that meet our criterion for all edges in E, we also ensure a covering of the edges, in the following sense: the value of the multiscale kernel Kij 

between xi and xj is identical to that of a kernel re-centered at the midpoint p ≡ (xi + xj )/2 and re-scaled using half the geometric mean of σi and σj as its scale, σp:

Kij = exp −‖ xi − xj ‖2

σiσj

= exp −‖ (xi − xj )/2‖2

σiσj /22 = exp −‖ (p − xi)‖2

σ2

> p

, (20) with σp ≡ √σiσj /2 (Figure 14). 24 C =1 C = 0.9 C =0.8 

> 0.0 0.5 1.0 1.5 2.0 Summation of Gaussians

Figure 13: Effect of hyperparameter C from equation 19 on the resulting weighted graph (left), optimal scales (middle), and manifold approximation (right, shown as the resulting summation over the Gaussian kernels around each point using their individual scales). For C = 1 (top), the scales overlap too much and, as a result, the Gaussian summation (right) is highly non-uniform. For 

C = 0.8 (bottom), the scales are not sufficiently large to properly cover the underlying manifold, resulting in holes (right). When C = 0.9, there is a good compromise between covering and keeping a uniform density, so the Gaussian summation approximates a partition of unity (summing to ∼1everywhere) when the scales correctly conform to the local sampling characteristics. Our approach will allow us to tune C based on a relaxation statistic, δ′

> i

.

Remark 8. We say a C-covering is attained when every pair (i, j ) ∈ E is C-connected (equa-tion 19). Additionally, when the spacing between neighboring points is approximately uniform locally, the pointwise summation over all Gaussian kernel bumps given by the individual scales provides an (un-normalized) partition of unity of M.We now use the covering constraints to solve for the set of scales, σ. It is desirable that the scales be small (respecting the reach), while at the same time maintaining the connectivity in G

close to that of G. Thus, one idea is to find scales such that the sum of edge weights in G incident to a node i from its neighbors in G approximate the degree of i in G, for all i, while at the same time ensuring a C-covering. This, however, amounts to a non-convex problem in which the cost function involves a summation of multiscale kernel values. We are unable to solve this efficiently. 25 ijrij rij rij 2 rij ¾irij             

> 2rij
> 23rij ¾j
> BAC
> ¾i¾j=r2
> ij ipj
> 01kernel value scale: p¾i¾j=212p¾i¾j

Figure 14: Covering constraint for the multiscale kernel of equation 13. Left: A graph G with two nodes i and j at a distance rij from each other in Rn. Since they are connected, their assigned individual scales σi and σj must satisfy σiσj ≥ r2 

> ij

, i.e., the covering constraint (here we assume 

C = 1). Center: All feasible pairs ( σi, σ j ) lie inside the region above a positive hyperbola, three of which are indicated as colored points; pairs A and B satisfy exactly, while C satisfies in excess. Each one is also depicted as a pair of circles on the left plot using the same color code, each one centered at its corresponding node (radii are set to half the scale, for clarity). Although pairs A and B differ in their ratio σi/σ j , both result in the same multiscale kernel value for the edge ( i, j ), since the product σiσj is the same; pair C yields a slightly higher value. This illustrates the freedom that might exist in choosing an optimal combination of scales for all nodes (i.e., a covering). Right: 

multiscale kernel values, Kij , centered at either i or j, shown in green, are symmetric (with scale 

√σiσj ). Horizontal axis represents position over the line in Rn passing through i and j. A kernel centered at the midpoint p between i and j using half the scale (black curve) attains the same value as Kij at i and j. Dashed red line indicates the common value between the three kernels. Instead, we find the smallest individual scales such that our covering criterion is satisfied for all edges (a “minimal covering”), and later address the quality of the relaxation by using a statistical pruning (edge sparsification). This can be transformed into a convex, linear program with linear constraints by which all scales can be solved for simultaneously, as we show next. (We also present, in Appendix A, a greedy approach to this optimization that may be convenient when dealing with very large datasets.) 

3.4.2 Linear program relaxation 

To achieve a minimal covering, one might minimize ∑ 

> i

σi (or, equivalently, the 1-norm of the vector σ, since scales are positive) subject to the covering constraint 3. This suggests the following:               

> 3Another possibility is to use a weighted sum ∑
> iνiσiwhile keeping the same constraints, thus still guaranteeing a covering. The weights νiadd a bias to how the length of an edge is split between its two incident nodes (by balancing their individual scales). One interesting option is to set νi=rnon
> i/r FN
> i, i.e., the ratio between the distance to the nearest non-neighboring point, rnon
> i, and the farthest neighbor, rFN
> i.

26 Optimization Problem: 

min  

> σ

1ᵀσ

s.t. (i, j ) is C-connected , ∀ (i, j ) ∈ Eσi is bounded, ∀ i ∈ V, 

(21) where σ is the vector of individual scales, σi, and 1 is the all-ones vector. Now it remains to represent the C-covering requirement by a set of constraints. Looking in detail at C-connectedness (equation 19) as a function of σi and σj , observe that it represents a region delimited by a single-branched hyperbola (since the distance and scales are positive): 

σiσj ≥ (Cr ij )2, σi > 0, σ j > 0, (22) where rij ≡ ‖ xi − xj ‖Rn ). Each σi is naturally bounded above by the distance to i’s farthest neighbor, rFN  

> i

:

σi ≤ rFN  

> i

, (23) beyond which all neighbors are satisfied 4, so further increasing either scale would make the weights to non-neighbors larger than strictly necessary (thereby hurting the kernel graph relaxation). These bounds, combined, specify a bounding box for each edge that must necessarily be crossed (or at least touched) by the hyperbola, since rij > 0.0 ui¾i

> 0
> uj¾jv

A  

> 0ui¾i
> 0
> uj¾jv

B 

> 0ui¾i
> 0
> uj¾jv

C  

> 0ui¾i
> 0
> uj¾jv

D   

> original constraint bounds convexified feasible region

Figure 15: Examples of constraints introduced by an edge, eij , in G. The C-connectivity rule, i.e., the hyperbola given by σiσj = ( C‖xi − xj ‖)2 (dashed curve), when convexified, may give rise to one or two linear constraints, depending on whether the hyperbola’s vertex v (point where σi = σj )intersects the bounding box given by the lines σi = 0 , σj = 0 , σi = ui, and σj = uj , where ui

and uj denote upper bounds. Hatched area (in orange) shows feasible region using convexified constraints; tangent line at v is shown in gray. When v is interior to the bounding box ( A), two secants (in blue) define the feasible region (namely, the lines passing through v and the points where the hyperbola intersects the lines σi = ui and σj = uj ); when either v = ui (B) or v = uj

(C), only one secant is necessary; when v coincides with both ui and uj (D) (which may occur if 

C is set to 1), again only one inequality is necessary, namely the tangent line at v.        

> 4That is assuming C≤1(a natural choice). If for some reason one needs to allow C > 1, then the upper bounds must be scaled by Cin order to ensure feasibility.

27 Due to the hyperbolae, this amounts to a non-linear, non-convex set of constraints. However, we can convexify the feasible set by considering, for each edge (i, j ), the line(s) passing through the hyperbola’s vertex (the point at which σi = σj = Cr ij ) and the points where the hyperbola intersects the bounding box. The four possibilities are shown in Figure 15. The feasible region for each edge, therefore, is bounded by a convex envelope given by such line(s) and those defined by the upper bounds to σi and to σj . Such envelopes for all edges, combined, define the boundaries of a convex polytope. Note that this convexification is conservative in the sense that only the objective is relaxed—the feasible scales are always at least as large as required by the original non-convex problem, therefore our covering requirement is not relaxed. (Because of the presence of a later pruning stage in the algorithm, it is better to over-connect here than to inadvertently disconnect nodes that should otherwise be connected.) Letting m ≤ 2|E| be the total number of linear constraints obtained as above, and N the number of nodes in G, we define the m × N matrix Λ and the m × 1 vector b. Now, for each edge, 

eij , let its two possible constraints be expressed as 

σj ≥ α(1)  

> ij

σi + β(1)  

> ij

(24) 

σj ≥ α(2)  

> ij

σi + β(2)  

> ij

(25) with αij and βij denoting, respectively, the slope and intercept of the corresponding line(s) forming its convex envelope. Rearranging, we obtain αij σi − σj ≤ − βij for each line, which is encoded as a row in Λ with values αij and −1 at columns i and j, respectively (with zeros everywhere else), and an entry in b with value −βij :

Λ

. . . i . . . j . . . 



... ... ... ... ...

e(1)  

> ij

0 . . . α(1)  

> ij

· · · 0 . . . −1 . . . 0

e(2)  

> ij

0 . . . α(2)  

> ij

· · · 0 . . . −1 . . . 0

... ... ... ... ...

m × N

σ

N × 1

≤

b



...

−β(1) 

> ij

−β(2) 

> ij

...

m × 1

.

Remark 9. The convex envelope defining the constraints can be expressed by the linear inqualities: 

Λσ ≤ b

0 < σ ≤ rFN , (26) where rFN is the vector of distances to each node’s farthest neighbor. Hence the problem now amounts to a convex, linear program (LP) with linear constraints: 

Optimization Problem: LP Relaxation: 

min  

> σ

1ᵀσ

s.t. Λσ ≤ b

0 < σ ≤ rFN ,

(27) 28 which can be readily solved by a variety of methods [see, e.g., 21]. Figures 19, 21, and 20 show the results of running this optimization on different examples. 

## 3.5 Sparsification 

Summarizing what we have seen so far, the Gabriel graph provides an initial estimate of connec-tivity, while the LP optimization provides minimal scales for a continuous kernel to cover those connections. However, since the initial estimate of the discrete graph might contain incorrect con-nections, its resulting optimal scales might also be inadequate. An example of this can be seen in Figure 19: initially, two pairs of nodes are connected across the central gap since a Gabriel ball exists between them. This will require very large scales to “cover” these edges. Furthermore, the Gabriel graph is based on a local connectivity rule; however, as illustrated in Figure 16, decisions about connecting nodes across a gap should not be local. We here address both of these issues, by introducing a global statistic based on how frequently such a gap occurs in the data. In terms of Algorithm 1, we are now at steps 6 and 7. 0.5 1.0 1.5 2.0 2.5 3.0 3.5 Volume ratio statistic, ±0i

> 010 20 30 40 50 60 70 80 Counts
> median threshold

Figure 16: Local vs. global assessment of neighborhoods. Left: The points inside the cropped win-dow appear to form two well-defined clusters when looked at up close (local estimation). However, when considered in the context of the full dataset (global estimation), the apparent gap between the top and bottom groups “disappears,” i.e., it is well within the range of gaps observed through-out the data. More precisely, it does not significantly deviate from the average sampling interval. 

Right: The converged graph G indeed connects the two groups by edges, and the distribution of volume ratios, δ′ 

> i

(lower inset), confirms that all edges are reasonable. 

3.5.1 Volume ratio 

Because incorrect connections can be given by Gabriel balls lying in the free space between parts of a manifold, i.e., across the medial axis, it is tempting to simply prune the longest connections. Note, however, that the size of a scale by itself is not necessarily important: in both examples shown in Figure 6, the non-uniform density causes scale sizes to vary considerably, and even the largest ones are appropriate, that is, are still consistent with the distances to neighboring points. 29 Conversely (and importantly), a scale that is excessively large will likely cover “too many” points. That is, it will cover neighbors in excess of the number of discrete neighbors of its cor-responding node in G. We quantify this notion by observing that an individual scale, σi, should produce kernel values whose sum is comparable to the discrete degree, deg( i), of node i in G. As will be shown, after proper normalization this also means σi shall relate to a local volume element around xi, or the inverse of the local density. Since each connection in G can be seen as having unit weight, a Gaussian kernel around xi with scale σi should distribute that same amount, deg( i),only continuously over ambient space. We start our derivation with a definition: 

Definition: let w(σi) 

> ij

be the Gaussian kernel value between xi and xj using scale σ(t)

> i

at iteration t. A (non-isolated) node’s volume ratio at iteration t, denoted by δ(t) 

> i

, is defined as 

δ(t) 

> i

≡

∑ 

> j∈V

w(σ(t) 

> i)
> ij

∑ 

> j∈V

aij 

, (28) i.e., the ratio between node i’s weighted degree due to σ(t) 

> i

and its discrete degree in G(t) (hence-forth we suppress the iteration dependency ( t) to simplify notation). An individual-scale Gaussian kernel is needed to correctly assess the impact of σi on the relaxation from the perspective of i

alone—the multiscale kernel here might artificially increase the weighted degree of i when other nodes (even non-neighbors of i!) have incorrect scales. (Nevertheless, as discussed below, a cor-responding ratio using the actual weights in G may eventually be used for convergence purposes.) Now, using a mean-value integral [as in 30], the numerator approximates the volume under the continuous Gaussian kernel over M, and can be further approximated by 

∑

> j

w(σi) 

> ij

≈ N

vol( M)

∫

> M

exp 

(−‖ xi − xj ‖2

σ2

> i

)

dxj (29) when M has uniform density and low curvature. In practice, the kernel will have compact support due to numerical precision (i.e., its values become effectively zero for sufficiently large distances), so by defining the volume element dV i ≡ vol( N (xi)) /|N (xi)| of a neighborhood N (xi) ∈ M 

around xi, we may rewrite equation 29 as 

∑

> j

w(σi) 

> ij

dV i ≈

∫

> M

exp 

(−‖ xi − xj ‖2

σ2

> i

)

dxj (30) when the sampling is approximately uniform around xi. By further assuming that σi is small, and that M can be well-approximated locally by its tangent space Rd, then 

∫

> M

exp 

(−‖ xi − xj ‖2

σ2

> i

)

dxj ≈

∫

> Rd

exp 

(−‖ xi − xj ‖2

σ2

> i

)

dxj = ( √πσ i)d, (31) so ∑

> j

w(σi) 

> ij

dV i ≈ (√πσ i)d, (32) as shown in Figure 17. 30 An analogous derivation for the discrete degree summation is as follows. First, note that the edge weight in this case is a constant (unity); it remains to determine its support over M. From section 3.3.3, we know that, for simple manifolds with random sampling, the node degree deg( i)

in a Gabriel graph is approximately 2di within a region of constant intrinsic dimensionality, where 

di denotes the local intrinsic dimension around xi (possibly different around other points in X )5.In more general manifolds, we expect the converged graph G? instead to approach such a property. This means ∑ 

> j

aij ≈ 2di will approximate the volume of a hyperrectangle (or box) of unit height and having a di-dimensional hypercube of side 2 as its base 6. So, by defining ρi as the radius of the local volume element dV i (such that ρi = di

√dV i), we may write: 

∑

> j

aij dV i ≈

∫ ρi

> −ρi

· · · 

∫ ρi

> −ρi

1dx j1 . . . dx jd = (2 ρi)di , (33) as illustrated in Figure 17. Hence, ρi is a kind of “neighborhood radius” of xi.From equations 32–33, equation 28 becomes 

∑ 

> j

w(σi)

> ij

∑ 

> j

aij 

=

∑ 

> j

w(σi) 

> ij

dV i

∑ 

> j

aij dV i

≈

(√πσ i

2ρi

)di

, (34) representing the ratio between the volume of a Gaussian with scale σi and that of a box of side 

2ρi and height 1 (cf. Figure 17). As the algorithm approaches convergence, we expect σi ≈ ρi

(scales are compatible with neighborhood radius) and deg( i) should approach, on average, the empirically-observed value of 2di (meaning that the number of neighbors in G is compatible with dimensionality of M). This results in 

∑ 

> j

w(σi)

> ij

∑ 

> j

aij 

≈

(√π

2

)di

. (35) Finally, we can estimate di as 

˜di ≡ log 2

∑

> j

aij , (36) based on the empirical degree distribution of G(t). From this, we can compute a normalized volume ratio, δ′(t) 

> i

, dividing δ(t) 

> i

by the value from equation 35: 

Definition: A node’s normalized volume ratio is computed as 

δ′(t) 

> i

≡

∑ 

> j

w(σi)

> ij

∑ 

> j

aij 

( 2

√π

) ˜di

. (37) Nodes whose degree deviate from exactly 2di will, likewise, under- or overestimate the local dimension, so reasonable volume estimates are still obtained regardless. However, in order to avoid        

> 5We abuse notation, therefore, when we say “ d-dimensional manifold”, or “ M ∈ Rd”.
> 6This agrees with our observation (section 3.3.3) that the unoccluded region around xiis similar to a di-orthoplex: by placing a vertex (i.e., a neighbor) in each of its 2difacets, we obtain a di-hypercube, which is the dual polytope of an orthoplex.

31 dimension less than 1 for connected nodes, in practice when deg( i) = 1 we replace ∑ 

> j

aij with 

max {2, ∑ 

> j

aij }.Thus, we expect δ′ 

> i

≈ 1 for points obeying σi ≈ ρi and ˜di ≈ di. Crucially, points for which these conditions are not met (those having “wrong” neighbors in the original Gabriel graph, G(0) )will depart from this by having δ′ 

> i

 1. In the next section, we shall use this fact to guide a sparsification of edges in G(0) based on δ′

> i

.vol-ratios 

Figure 17: Computing the volume ratio between continuous and discrete degrees of a node i with neighboring points sampled uniformly over a d-dimensional manifold M. Top row: Using a Gaus-sian kernel, the weighted degree of i (sum of kernel values ∑ 

> j

w(σi) 

> ij

) in G approximates the volume of a Gaussian with scale σi (equation 32). Bottom row: The number of edges adjacent to i in G

(sum of unit weights) approximates the volume of a box with unit height and a hypercube of side 

2ρi as its base, where ρi is the radius of a local volume element of M around xi (equation 33). 

Right: When the scale σi is compatible with ρi, the volume ratio, δi, is expected to be approxi-mately (√π/ 2) d, and therefore is a scale-invariant quantity. Interestingly, δ′ 

> i

can also be interpreted as measuring how well the scale σi fits the local volume element dV i (or, equivalently, how it counteracts the local sampling density, 1/dV i). Since dV i =

ρdi 

> i

(from the definition of ρi), we may rewrite equation 34 as: 

∑ 

> j

w(σi)

> ij

∑ 

> j

aij 

≈ (√πσ i)di

2di dV i

. (38) Summarizing the above, when ˜di ≈ di and σi ≈ ρi we have: 

Remark 10. A node’s normalized volume ratio may alternatively be expressed as 

δ′(t) 

> i

≡

∑ 

> j

w(σi)

> ij

∑ 

> j

aij 

( 2

√π

) ˜di

≈ (√πσ i)di

2di dV i

( 2

√π

) ˜di

≈ σdi

> i

dV i

. (39) Therefore, δ′ 

> i

can be thought of as the product between kernel scale and local density. When σi is optimal, it should be approximately equal to the inverse of the local density, so δ′ 

> i

≈ 1.32 3.5.2 Uniformity of sampling and edge pruning 

Since δ′(t) 

> i

is evaluated for every node xi, we can collect it across nodes and view it as a statistic. This has two consequences: ( i) it can be used to enforce consistency in sampling, and ( ii ) outliers in this statistic are likely candidates for edge pruning. We address consistency of sampling first. We have several times stated that sampling is required to be locally uniform, although its rate may change over the manifold. Examples of this were shown in, e.g., Figure 12, where the sam-pling was denser in the center of the Gaussian distribution than in the periphery. This example differs from the regular grids, in which all nearest neighbors had exactly the same distance. Putting this together, we have: 

Remark 11. Locally Uniform Sampling: Let node i have ki neighbors in G(t). Among these, let 

rFN  

> i

denote the distance from xi to its farthest neighbor, and rNN  

> i

that to its nearest neighbor. When 

rFN  

> i

≈ rNN  

> i

for all i, we say the sampling is locally uniform. This is useful because a departure from the assumption that sampling is locally uniform will cause δ′ 

> i

to be on average greater than 1 throughout the dataset. To see this, when sampling is not uniform, we have rFN  

> i

> r NN  

> i

. Now, since σi is optimized to cover all of i’s neighbors, it will have in most cases the same order of magnitude as rFN  

> i

(minus some possible slack due to the multiscale interaction). Therefore, the higher the variability in the neighbors’ distances, the larger the difference between rFN  

> i

and rNN  

> i

will be, making σi, in turn, be larger than the distance to most neighbors of i. Ultimately, this will increase ∑ 

> j

w(σi) 

> ij

beyond what we would have in a uniform-sampling scenario (in which rFN  

> i

≈ rNN  

> i

). When data are acquired using a global sampling strategy, this variability in the neighbors’ distances should be roughly constant throughout the dataset (rather than the distances). So we use the scalar parameter, C, from equation 19 to correct for this “bias” and bring the median of the distribution of δ′(t) 

> i

(denoted as 〈δ′(t) 

> i

〉) close to 1 7.

Remark 12. Let the tuned C?(t) be that which causes 〈δ′(t) 

> i

〉 to be closest to 1. Typically, C?(t) < 1, which, in the scale optimization procedure, means that the covering constraints (equation 22) are being relaxed using the distribution of δ′ 

> i

as a guide (Figure 18). Note that, although the tuning of C is not necessary for finding candidates for sparsification, it attributes a quantitative meaning to the value of δ′

> i

, so any δ′ 

> i

 1 is guaranteed to indicate the need for edge pruning. Such tuning should be performed at t = 0, and repeated as needed over the iterations whenever 〈δ′(t) 

> i

〉 deviates too much from unity (which may happen after several edges have been pruned). Most commonly, we find 0.5 < C ?(t) < 1.Thus, we have a data-driven way of finding an appropriate value for C. Because it is a global constant applied to all connection constraints, it shifts the distribution of δ′ 

> i

to have median around 1 without changing its general shape. This leads us to the second use of our statistic: any node whose normalized volume ratio is much greater than the median of the population should be identified as an outlier. Such nodes will have a neighbor considerably farther than its other neighbors (relative to the median variability of such neighboring distances throughout the data), and are candidates for the sparsification step.   

> 7Although the mean typically gives smoother tuning curves, the median is more robust. This matters, because of the possible outlying δ′
> ivalues.

33 G⋆      

> 0.70 0.75 0.80 0.85 0.90 0.95 1.00

C

> 0.7 0.8 0.9 1.0 1.1 1.2
> ± 0
> i
> ®

C⋆ = 0.915 

## G⋆  

> 0.5 1.0 1.5

±0i   

> 010 20 Node count  C=0.800
> 0.5 1.0 1.5

±0i   

> 010 20
> C=0.915
> 0.5 1.0 1.5

±0i 

> 010 20
> C=1.00
> median
> ®

Figure 18: Tuning the hyperparameter C based on the median of the distribution of normalized volume ratios, 〈δ′

> i

〉. Left: Converged unweighted graph G? obtained for the dataset from Figure 19. 

Center: After computing 〈δ′

> i

〉 for a range of values of C ≤ 1, the optimal C? is that resulting in a 

〈δ′

> i

〉 closest to 1. Histograms below show the distribution of δ′ 

> i

for different values of C, including 

C? = 0.915. Right: The resulting weighted graph G? after C-tuning typically exhibits a more uniform connectivity throughout (see Figure 13). 

Remark 13. Nodes that are robust outliers according to the δ′ 

> i

statistic have an overly distant neigh-bor (relative to the other neighbors for that node) and hence are likely to be in violation of reach or other geometric constraints. These relatively distant neighbors are candidates for having an edge pruned. Given the distribution of normalized volume ratios, statistical models can be used to define a threshold for identifying outliers (see Figures 19–22). It is likely that datasets with a large number of problematic connections will exhibit a distribution with a heavy tail, or that looks like a mixture of two distributions (cf. example in Figure 22), so using the distribution’s quartiles may give a more robust result. One option that seems to work particularly well is to use estimates of the sample mean and standard deviation from the quartiles, as in [105] (throughout, we make use of the C3 method derived therein, setting the δ′ 

> i

threshold to 4.5 standard deviations above the mean thus estimated). Still, we found that results are typically quite invariant to this particular choice, especially in real-life datasets. Finally, we note that our algorithm can be run interactively, so the user can analyze the histogram of the distribution after each iteration to judge whether the choice of threshold is reasonable and thus be confident in the results. Nodes with δ′ 

> i

above the threshold should have their connection to their farthest neighbor deleted. Ideally, only one such connection is pruned after each iteration; however, should that 34 become impractical with large datasets, a compromise is to limit the pruning, at each iteration, to a single edge from each node that is above the threshold (giving the chance for its δ′ 

> i

value to be updated before the next pruning). 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 

± 0                 

> i(0)
> 0510 15 20 25 Node count Volume ratios, t= 0
> threshold ¾(0)
> Individual scales, t= 0
> G(1)
> Edge sparsification, t= 0 ¡1
> 0.50 0.75 1.00 1.25 1.50 1.75 2.00

± 0    

> i(2)
> 0510 15 20 25 Node count Volume ratios, t= 2
> threshold

¾⋆   

> Individual scales, t= 2
> G⋆
> Converged

Figure 19: Optimal scales and associated normalized volume ratios at iterations 0 and 2 of Algo-rithm 1 on the horseshoe dataset (see Figure 13). Top row: the δ′ 

> i

statistic has median around 1 and several outliers. These are caused by the long edges and huge scales (middle). Right column: G(t)

after iteration 1, with edges deleted shown in red (top), and after iteration 2 (bottom). 

3.5.3 Convergence 

The algorithm converges at iteration t when no point i has an outlying δ′(t) 

> i

(i.e., greater than a statistical threshold). This implies that no edges will be pruned, so G(t+1) = G(t) and therefore no further changes can occur to either σ(t) or G(t). Note that convergence is guaranteed: since at every iteration t an edge must be removed, the algorithm necessarily reaches a certain t at which all outliers (if there were any to begin with) have been pruned. If one is solely interested in obtaining G? (i.e., not interested in G?), an alternative convergence condition may be adopted that looks at the distribution of the (normalized) multiscale volume ratio ,

δ′(t) 

> iMS

:

δ′(t) 

> iMS

≡

∑ 

> j

wij 

∑ 

> j

aij 

( 2

√π

) ˜di

, (40) 35 0 5 10 15 20 

± 0    

> i(0)
> 0510 15 20 25 30 35 40 Node count Volume ratios, t= 0
> threshold

¾(0)   

> Individual scales, t= 0

G (2)            

> Edge sparsification, t= 0 ¡2
> 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0

± 0    

> i(3)
> 010 20 30 40 Node count Volume ratios, t= 3
> threshold

¾(3)   

> Individual scales, t= 3

G (5)          

> Edge sparsification, t= 3 ¡5
> 0.6 0.8 1.0 1.2 1.4 1.6

± 0    

> i(6)
> 0510 15 20 25 30 35 Node count Volume ratios, t= 6
> threshold

¾⋆  

> Individual scales, t= 6

G ⋆

> Converged

Figure 20: Optimal scales and associated normalized volume ratios at different iterations of Al-gorithm 1 on the dataset from Figure 6. The distribution of δ′ 

> i

(left) indicates those connections that are least likely to represent reasonable geodesics over the underlying manifold. Right column shows G(t) after iterations 2, 5 and 6 (deleted edges in red). analogous to equation 37 but using the weights from G(t) directly. Since the multiscale kernel takes into account the interaction between individual scales, the distribution of δ′ 

> iMS

will be typically tighter than that of δ′ 

> i

(i.e., some of the excessively large scales might be compensated by small neighboring scales). Therefore, one may wish to allow for an earlier convergence when there are no remaining outliers in the distribution of δ′ 

> iMS

.Finally, in applications where it is required that G? be connected, pruning can simply be 36 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 

± 0    

> i(0)
> 010 20 30 40 Node count Volume ratios, t= 0
> threshold

¾(0)   

> Individual scales, t= 0

G (2)            

> Edge sparsification, t= 0 ¡2
> 0246810 12 14

± 0    

> i(3)
> 010 20 30 40 Node count Volume ratios, t= 3
> threshold

¾(3)   

> Individual scales, t= 3

G (6)           

> Edge sparsification, t= 3 ¡6
> 0.5 1.0 1.5 2.0 2.5 3.0 3.5

± 0    

> i(7)
> 010 20 30 40 Node count Volume ratios, t= 7
> threshold

¾⋆  

> Individual scales, t= 7

G ⋆

> Converged

Figure 21: Optimal scales and associated volume ratio statistics at different iterations of Algo-rithm 1 on the clustered dataset from Figure 6. Pruned edges (in red) are precisely those connecting the three clusters together. stopped before disconnection. Naturally, G? is always connected up to machine precision or some numerical tolerance. In closing this section, we return to one of our introductory examples and show, in Figure 23, the resulting graphs for the sampling Swiss cheese patterns (from Figure 2). When sampling is too sparse (bottom), there is only so much that can be inferred, and not all holes are free of edges after convergence. As sampling gets denser, however, the algorithm correctly identifies that edges across holes should be pruned (middle). When it is very dense (top), even the initial Gabriel graph 37 0 10 20 30 40 50 

± 0 

> i(0)
> 0100 200 300 400 500 600
> Node count

Volume ratios, t = 0 

> threshold

¾(0) 

Individual scales, t = 0 

G (136) 

Edge sparsification, t = 0 ¡ 136      

> 0510 15 20 25

± 0 

> i(137)
> 050 100 150 200 250
> Node count

Volume ratios, t = 137 

> threshold

¾(137) 

Individual scales, t = 137 

G (273) 

Edge sparsification, t = 137 ¡ 273     

> 0.5 1.0 1.5 2.0 2.5

± 0 

> i(274)
> 020 40 60 80 100 120
> Node count

Volume ratios, t = 274 

> threshold

¾⋆

Individual scales, t = 274 

G ⋆

Converged Figure 22: Optimal scales and associated normalized volume ratios δ′

i after each iteration of the algorithm on the dataset from Figure 28 (here, seen from a lateral view). The number of initial connections in G(0) (Gabriel graph) is very large, so the initial distribution of δ′

i shows two modes. However, ratios in right-side peak are very high, and are therefore easily identified as outliers. The algorithm converges soon after all edges crossing the gap are eliminated. is able to correctly infer the true holes. 38 X G ⋆ G⋆      

> XG⋆G⋆
> XG⋆G⋆

Figure 23: Sampled Swiss cheese results (cf. Figure 2). The original sampled points (true holes outlined) are shown, together with the converged graphs. In the sparse case (bottom), sampling is close to locally uniform so not all holes are correctly inferred. As sampling gets denser (top two rows), no holes are violated. 

## 3.6 Comparison with other kernel methods 

We now compare the data graphs obtained using our iterated adaptive neighborhoods (IAN) with those from other popular manifold learning methods. In Figure 24, a synthetic “stingray” dataset exhibits a transition of apparent dimension from 2 (body) to 1 (tail), a variation of the scenario explored in Figure 4. Points were uniformly sampled, with 20% deleted at random. 39 Our converged, unweighted graph, G? (top row in Figure 24), can be compared with the tra-ditional k-nearest neighbors graph (bottom row), used in a variety of methods, including Isomap [96]. In the latter, when k = 2, the tail exhibits perfect connectivity, but the body is too sparse. If 

k = 4, the body is more properly connected but the tail becomes overly connected, and “folding”, or “short-circuits”, start to appear. Finally, for k ≥ 8, the connectivity is inappropriate as the tip of the tail connects directly to the body. In contrast, G? manages to retain a minimally-connected tail while covering the body almost everywhere, creating appropriate edges across many of the sampling gaps (compare with the holes that remain in the k-NN graph with k = 4, some of which are present even when k = 8). Our weighted graph, G?, can be compared against methods that use a Gaussian-like kernel, and where each point has an individual scale. Some of these methods were described in section 2.1: t-SNE [101], UMAP [77], self-tuning [111], and variable bandwidth [17, 16]; their resulting con-nectivity can be visualized in Figure 24, where edges have intensity proportional to their weight. In Figure 25, we visualize the individual scales resultant from each of these methods. Each σi

is represented, around each point i, as the level set corresponding to a (single-scale) kernel value of 0.75. At the top, we see that the scales found by our kernel seem to nicely conform to the space between each point and its neighbors. Especially illuminating is what happens along the tail, where scales either “expand” or “shrink” so as to minimally cover the spaces between neighboring points; this illustrates what our scale optimization achieves. Among the other methods, with few exceptions, the scales seem to cover either too much (collapsing the tail on itself) or too little (leaving holes in the body). The weighted graphs in Figure 24 reveal the result of the interaction between these individual scales (namely, the edge weights). Our G? (top right) manages to cover almost the entire body with edges, while keeping the tail minimally connected—in fact, resembling the unweighted version in 

G?, and therefore respecting the original curvature and reach. Other methods, in contrast, have a hard time achieving both things with a global value for k. In t-SNE, the scales over the body are much small when k ≤ 4, so its weighted graph looks too sparse; for k ≥ 8, the scales over the tail become too large, and therefore strong edges appear, connecting it to the body. In UMAP, the scales do not grow as much with increasing k, but at k = 4 the body in the weighted graph is still too sparse, while for k ≥ 8 the tail is strongly connected to the body. With the self-tuning, scales seem to grow faster with k, while with variable bandwidth this growth is somewhat counteracted by the action of their global scale,  (equation 7). In fact, the graph that most resembles our own 

G? is the one using the variable bandwidth kernel with k = 2, the main difference being that the big sampling gap near the tip of the body is poorly connected, while in our case it is slightly overly connected (due to connections in G? crossing that gap). 

# 4 Applications 

We now provide examples of application of our kernel to three different manifold learning tasks: dimensionality reduction, by means of a non-linear embedding algorithm; geodesic estimation, which typically finds application in computational geometry, vision, and graphics; and local in-trinsic dimensionality estimation. 40 wgraphs-ok-2 Figure 24: Top: The stingray dataset and the converged graphs, G? and G?; pruned edges are shown in red. Bottom: Other algorithms produce qualitatively different graphs depending on the neighborhood size parameter, k. All graphs shown are weighted (using a continuous kernel) except for the k-nearest neighbors graph (bottom row). Edge weights are visualized as the intensity of the line segments (each wij is divided by the kernel value when rij equals the scale, for a fair comparison across algorithms). 41 Figure 25: Individual scales obtained using our algorithm (top) compared to other methods (bottom table), as represented by their level sets for a (single-scale) kernel value of 0.75. 

## 4.1 Low-dimensional embeddings 

Dimensionality reduction is now ubiquitous in visualization of high-dimensional data. Several methods exist [92, 69, 50, 102], and most of the non-linear methods are manifold-based [88, 96, 87, 57, 10, 36, 112, 28, 109, 101, 95, 77, 80]. Given a collection of points in high-dimensional space sampled from a low-dimensional manifold M, the goal is to find a good parametrization for the data in terms of intrinsic coordinates over M, which in turn can be used to produce a 42 low-dimensional embedding. In surveying the literature, it is common to find a heuristic, or a range of values, suggested for choosing the neighborhood size (see section 2.1), but rarely do we see examples of the sensitivity of the results to that choice. In this section, we ran a few of the most popular methods using a wide range of values for the kernel scale parameter, k, and compared their results to those using our own kernel. We have limited our comparison to some of the embedding methods that use a neighborhood kernel and for which pairwise information is sufficient as input (i.e., do not require positional information): diffusion maps [29, 28], Isomap [96], t-SNE [101], and UMAP [77]. As shown in Figures 26–28, results can vary qualitatively depending on the choice of k. Five values were tested for each dataset, spanning a wide range of scales and different geometries. Next, we summarize each of these methods and their results. 

4.1.1 Diffusion maps + self-tuning kernel 

Diffusion maps are based on the spectral properties of the random walk matrix (normalized graph Laplacian) over the weighted data graph; integration over all paths in the graph makes diffusion distances, in principle, more robust to “short-circuiting” than graph geodesics. For better compari-son with IAN, instead of the standard single-scale Gaussian kernel we use the self-tuning approach of Zelnik-Manor and Perona [111] from equation 6. Our kernel was applied to diffusion maps by directly using G? as similarity matrix (weighted adjacency matrix). We use the diffusion map parameters α = 1 and t = 1 [cf. 28]. With the stingray dataset (Figure 26), we see that the fully-extended tail at k = 2 becomes progressively more folded and compressed as k increases. The body appears contracted at k = 2, but expands with larger k. Using our own G?, although we obtain excellent embeddings of both body and tail (right-most column), they are represented by separate sets of coordinates (two for the body, and a third for the tail), which happens due to the change in dimensionality. Applying self-tuning to the spiral dataset (Figure 27), only k = 2 and k = 4 were able to prevent folding. The bent plane (Figure 28) was more tolerant, with good results for all k except 64, for which the plane remained folded. When using IAN, a good parametrization was obtained for both datasets. 

4.1.2 Variable bandwidth diffusion embedding 

We also tested a variant of diffusion maps using the variable bandwidth kernel of Berry et al. [17], in which a distinct type of multiscale kernel is proposed, along with a specific normalization of the weighted graph Laplacian. Because it computes an additional global scale, , based on the individual scales, in order to apply our algorithm to this method we replaced the density estimates, 

q (equation 7), with the inverse of our optimal scales. We used α = 0 and β = -1/2, as recommended in [16]; eigenvectors were scaled by the square-root of the inverse of their respective eigenvalues [90, 84], following the implementation in [8]. This method produced good embeddings for the stingray, especially for k = 8 (Figure 26). For the spiral (Figure 27), using k ≤ 8 caused some points to drift apart, and although it returned basically the original curve when k = 16 or 32, a spectral algorithm such as this is expected to “unroll” the spiral, finding a good (1-D) parametrization of it. The same happened with the bent 43 grid-embeds Figure 26: Running different embedding algorithms on the “stingray” dataset (see Figure 24). Different choices of the neighborhood size, k, may produce qualitatively different results, depend-ing on the algorithm. Running those same algorithms using the IAN kernel (right) typically gives a reasonable result. Refer to main text for details. plane (Figure 28), which could not be embedded into 2 coordinates for any choice of k. Using our scales, however, the algorithm managed to find appropriate parametrizations for all three datasets. 

4.1.3 Isomap 

Isomap applies classical multidimensional scaling (MDS) to geodesic distances computed as short-est paths over a k-nearest neighbors graph (equation 2). Because the graph is unweighted, this method is particularly sensitive to the choice of k. Our kernel was applied to Isomap by directly replacing the k-NN graph with G?.With the stingray (Figure 26), Isomap produced a good embedding with k = 4. The result 44 with k = 2 was completely wrong (an additional tail appears), and with k = 8 the tip of the tail was disconnected. With k = 16 and k = 32, it essentially returned the original data, without any dimensionality reduction. Our G? improved on the result of k = 4 by making the points in the body more uniformly spread. The spiral (Figure 27) was properly embedded (1-dimensional) only when k ≤ 4. With the bent plane (Figure 28), good results were obtained for k between 4 and 16, but k = 2 produced 1-dimensional curves, and k = 64 did not completely unfold it. Our G? produced the correct mapping in either case. 

4.1.4 t-SNE and UMAP 

t-SNE and UMAP are related methods that have gained popularity in recent years [9]. Both com-pute similarities between data points using individual scales based on log 2 k (section 2.1), and adopt a secondary kernel for computing similarities between embedded points: t-SNE uses a Student t-distribution (Cauchy kernel), while UMAP uses an non-normalized variant requiring a hyperparameter, min dist . In t-SNE, embedding coordinates are initialized at random, while UMAP adopts the strategy of refining an initial spectral embedding. Both then optimize their embeddings by running gradient descent on an information-theoretic cost function between sim-ilarities in input space vs. embedded space: t-SNE minimizes the KL-divergence; UMAP uses a variant of cross-entropy. Alternative initializations are typically used with t-SNE (e.g., PCA) to improve results [63, 72, 64]; in our experiments, for better comparison with UMAP, we used a spectral embedding initial-ization computed from its own symmetrized similarity matrix (equation 10). The IAN kernel was applied to t-SNE by replacing the individual scales (equation 8) with those in σ?; with UMAP, be-cause a different kernel function is used, we directly replaced the weighted graph (with adjacencies given by Uij in equation 12) with G?.We executed t-SNE assigning the various k values to the perplexity parameter, leaving the remaining parameters to their defaults in the scikit-learn implementation [85]. We used the Barnes-Hut method [100] for the cylinder dataset; and the “exact” method for all others. In UMAP, the n neighbors parameter was set to k, with remaining parameters using default values (in particular, min dist = 0.1). Because of the stochastic nature of both algorithms (even when using a fixed initialization), different runs will produce slightly different results. Therefore, in order to avoid “cherry-picking”, both algorithms were executed a single time, using the same random seed. Results for the stingray (Figure 26) were quite analogous between the two algorithms: both produced artificial clustering for k ≤ 8, while for k ≥ 16 the tail began to fuse with the body. The gaps in sampling within the body were accentuated by both algorithms, even at k = 32, where we see a big hole in the UMAP embedding; in t-SNE, it almost breaks into two pieces (despite the large neighborhood size). This example is illustrative of how much an embedding algorithm based on attractive vs. repulsive forces can end up exaggerating nonuniform sampling. The spiral (Figure 27) was disconnected by t-SNE for all values of k except 8. UMAP produced reasonable results for k between 4 and 8; however, for k = 2 a multitude of clusters was obtained, and when k ≥ 16 the curve twisted over itself. Using our kernel (right column) produced a con-nected, non-self-intersecting curve. Neither algorithm was capable of returning a good arc-length parametrization of the spiral, however. With the bent plane (Figure 28), although both algorithms succeeded in unfolding it, t-SNE 45 was only able to produce a fully two-dimensional plane (with no gaps) when setting k = 32 (not shown) or 64, while UMAP required k ≥ 16. Both gave reasonable results using our kernel. Figure27 grid-embeds-spiral 

Figure 27: Running different embedding algorithms on the spiral dataset (top), in which points are sampled from a unit-speed parametrized Archimedean spiral. Different choices of the neigh-borhood size, k, may produce qualitatively different results, depending on the algorithm. Running those same algorithms using the IAN kernel (right) typically gives a reasonable result. Refer to main text for details. 

4.1.5 A higher dimensional example 

Because all of the examples above have d ≤ 2, we also tested our kernel when applied to a higher dimensional manifold, namely a 5-dimensional cylinder ( R1 × S4) with radius 1 and length 3, 46 Figure28 grid-embeds-plane Figure 28: Running different embedding algorithms on the bent plane dataset (top), generated by extending a unit-speed parametrized catenary curve into two dimensions. Different choices of the neighborhood size, k, may produce qualitatively different results, depending on the algorithm. Running those same algorithms using the IAN kernel (right) typically gives a reasonable result. Refer to main text for details. sampled uniformly at random ( N = 8403, ambient space R6). On the other hand, here we used a pure, connected manifold with no bottlenecks and low curvature in order to simplify interpretation. Figure 29 shows two-dimensional embeddings obtained by applying our kernel to different embedding algorithms. Although all correctly produced an oblong, various degrees of mixing of the original color labels were observed, which can be used to qualitatively indicate the quality of the embedding schemes. A quantitative assessment was computed as the rank correlation coefficient, 47 or Kendall’s tau [60, 62] between the ranking (positional order) of each point along the main axis in the original vs. embedded spaces. Both t-SNE and UMAP produced similar or better results when using the IAN kernel (we set 

k = 27 based on the mean degree found in G?, compatible with d = 5; results were robust to this particular choice). Despite their current popularity [e.g., 108, 7, 27, 35, 9, 63, 46, 64, 107], produced considerably jittered outputs, however, implying that the original neighborhoods were not preserved. This appears to be caused by an attempt to reproduce the spherical shape of the cylinder’s base along the main axis, so different “slices” ended up projected on top of one another. However, UMAP produced jittered results even when set to return 6 components (as in the original space) instead of 2. Diffusion maps using IAN resulted in little mixing except near the boundaries, so neighbor-hoods were better preserved. Running it with either self-tuning or variable-bandwidth kernels using k = 27 gave comparable results; Isomap also produced excellent results, with tau = 0.98 (not shown). 48 cylinders Figure 29: Performance of different embedding algorithms on a 5-dimensional cylinder ( R1 ×

S4) sampled uniformly at random ( N = 8403, ambient space R6). Top left: original data, X ,projected onto first 2 coordinates (points colored according to their position along the cylinder’s long axis). Other plots show embeddings using different kernels and/or algorithms. The resulting degree of mixing of the original color labels indicates the quality of the embedding. A quantitative assessment (plots to the left of each embedding) was computed as the rank correlation coefficient, tau (see main text), between the ranking (positional order) of each point along the horizontal axis in the original vs. embedded spaces (a value closer to 1 indicates fewer exchanges in the original order). Use of the IAN kernel produced similar or better results with both t-SNE and UMAP ( k =27 was set based on the mean degree of G?, compatible with d = 5). Diffusion maps resulted in very little mixing except near the boundaries. 

## 4.2 Geodesic computation 

Using the unweighted graph, G?, one may immediately compute graph geodesics (shortest paths using distances in ambient space as edge lengths) to estimate the true geodesics over M [96]. The latter are likely to be underestimated by the former when sampling is sparse [15], even when the graph connectivity is correct, e.g., due to curvature (cf. section 3.3.2). It seems a good idea, then, to incorporate the continuous kernel values present in its weighted counterpart, G?, as a means to possibly improve geodesic estimation. We propose to use the heat method for geodesic computation of Crane et al. [32]. It consists in solving the Poisson equation to find a function, φ, whose gradient follows a unit vector field, X,pointing along geodesics; X can be obtained by normalizing the temperature gradient, ∇u, due to a diffusion process in which heat, u, is allowed to diffuse for a short time. Although this method is 49 tailored to applications where positional information and dimensionality are known (in particular, surfaces in R3), here we apply it to G?, since discrete versions of the operators used (Laplacian, gradient, and divergence) can be readily defined on a weighted graph [see 34]. Despite using pairwise information only, our method produces reasonable estimates, as shown in Figures 30 and 31. To understand why, notice that IAN indirectly solves for a weighted graph for which a random walk starting at node i has a higher probability of reaching a node in its discrete neighborhood, N (i), than any other non-neighboring node. Given that random walks are closely related to diffusion over a graph, one should expect G? to be able to provide reasonable information about how a diffusion process propagates over M. In other words, the Laplacian obtained from G?

should be a good approximation of a continuous operator over M—this is empirically confirmed by our results. In Figure 30, heat geodesics computed from G? for the bent plane dataset approximate well the true geodesics over M, and graph geodesics obtained from G? follow closely. Comparison with those from a naive k-NN graph illustrates that the choice of k is critical (compare with the bottom row of Figure 28). In Figure 31, we compare the results using weighted graphs from various kernels on the stingray dataset; interestingly, heat geodesics computed from G? hold reasonably well even when facing a continuous change in dimensionality. (The diffusion time parameter used by the heat method was optimized for each dataset.) 50 bentplane-geo Figure 30: Geodesic estimation for the bent plane from Figure 28; yellow points are closer to the source (marked with an arrow in the ground truth plot). Top: different views of the data in 3-D, with points colored according to the heat geodesics computed from G?. Middle: Geodesics displayed on an unbent version of the dataset: heat geodesics approximate well the true geodesics over M,and graph geodesics computed from G? follow closely. Bottom: graph geodesics computed from 

k-NN graphs using different choices of k; choosing k = 16 gives near-perfect results, but k = 4 shows distortions, and k = 66 misses completely. 51 Figure31 grid-stingray-geo Figure 31: Geodesics estimated using the heat method applied to G? are close to the ground truth (top). Other kernels yield suboptimal results for most choices of k (bottom); in particular, notice how the tip of the tail is usually inferred to be closer than it should (due to its being directly connected to the body in the underlying graph, cf. Figure 24). Yellow points are closer to the source (marked with an arrow in the ground truth plot). 52 4.3 Local dimensionality estimation 

Intrinsic dimensionality (ID) estimation is tightly associated with dimensionality reduction tasks, especially in manifold learning, where knowledge of d can help, among others, to determine the appropriate number of embedding dimensions. Informally, ID may be seen as the minimum num-ber of parameters required to accurately describe the data. In the context of manifold learning, it is typically equivalent to the topological dimension of M (e.g., a general space curve has dimen-sionality 1 since it requires a single parameter, arc length). There are many different ways to estimate it [24, 25]; global approaches are typically divided into two. The first group is based on some variant of PCA [e.g., 47, 73], and use the number of significant eigenvalues to infer dimensionality; these may be applied globally or by combining local estimates. The second group of methods, termed geometric (or fractal, when a non-integer ID is computed), exploit the geometric relationships in the data, such as neighboring distances. Some are based on estimating packing numbers [58] or on distances to nearest-neighbors [99, 86, 104, 31, 39, 18]. Among the most popular are the correlation dimension methods [26, 51, 55], a variant of which has been specifically applied in the context of determining an appropriate kernel width for manifold learning [see 30, 17, 52]. The dimension is computed as the slope of a log-log plot of the number of neighboring points vs. neighborhood radius (see section 2.1). A recent variation is [61]; others cover the difficult case of high ID [26, 89]. In our scenario, since we do not assume a pure manifold (section 3.1), we focus on local (i.e., pointwise) ID estimation approaches, namely those in which dimension is estimated within a neighborhood around each data point [e.g., 40, 54]. This notion can be formalized as the local Hausdorff dimension [110, 25], and a global estimate is typically found by averaging over local values. A popular approach is the maximum likelihood estimator (MLE) of Levina and Bickel [70], which computes local dimension based on k-nearest neighbors: 

ˆmk(xi) = 

(

1

k − 1

> k

∑

> j=1

log Tk(xi)

Tj (xi)

)

(41) where Tj (xi) denotes the distance between xi and its jth nearest neighbor. We shall use this method in our experiments, in which we compute a final mk(xi) by averaging ˆmk(xi) over i’s neighbors in order to reduce the variance of the local estimates (in the original, this is done over all data points). Notice that our kernel can be readily used with this method by simply replacing the k-NN graph with G?, therefore summing over nodes in the neighborhood N (i) instead of over the k nearest. Additionally, we propose a correlation dimension-based method that allows for local estimates. We describe it next, then compare its results with those from the MLE method. 

4.3.1 Algorithm: Neighborhood Correlation Dimension 

Our proposed method is adapted from the approach from Hein and Audibert [55] [also used in 30, 52, 17], where an estimate of correlation dimension is obtained using a general kernel. It consists in computing a curve, Z(σ), over all pairwise kernel values (e.g., a Gaussian) at different 53 values of the scale parameter σ:

Z =

> N

∑

> i=1
> N

∑

> j=1

exp − ‖ xi − xj ‖2

2σ2 . (42) As in [30] (and analogous to equations 29–31), by assuming that for small values of σ the manifold 

M looks locally like its tangent space, Rd, we have 

Z ≈ N 2(√2πσ )d

vol 2(M) , (43) which, after taking the logarithm, yields 

log Z ≈ d log σ + log N 2(2 π)d/ 2

vol( M) , (44) so the slope of log Z × log σ can be used to estimate the global dimensionality of the manifold, d.To do so, one typically looks for a region where this slope is most stable, i.e., the curve is approx-imately linear. Automated ways of finding the slope of such a region are: by linear regression of the middle portion of the curve [55] or by taking a point of maximum of Z′(σ) [17, 52]. However, because we assume that intrinsic dimension may vary over M, global averages can-not work in general. Moreover, nonuniform density, curvature, or multiple connected components may all create multiple peaks for Z′(σ), so inspection of the log-log plot cannot be automated. Therefore, we modify this approach to use individual Zi(σ) curves for each data point xi. To keep the summation local, points are restricted to those in the neighborhood of i in G?. Here, it is advantageous to work with an extended neighborhood (e.g., by also including neighbors-of-neighbors) due to the theoretical limit to the value of the dimension d that can be accurately estimated given a set of N points [38], namely d < 2 log 10 N . In fact, if N is large compared to d, even additional hops away from i may be considered. Because such extension is done by following edges in G? (as opposed to naively expanding a ball in Rn), we may thus obtain a larger (approximately tubular) neighborhood around xi without ever leaving the manifold. We denote such a neighborhood N ′(i), as opposed to the immediate neighborhood N (i); throughout this section, both will include the node i itself. Our algorithm involves the following steps: 1. For each data point xi and its extended neighborhood, N ′(i), define Zi as 

Zi(σ) = ∑ 

> j∈|N ′(i)|

exp − ‖ xi − xj ‖2

2σ2 . (45) 2. Analogous to equation 44, by taking the logarithm we have that the slope of the log Zi ×log σ

curve, i.e., 

Z′

> i

(σ) def 

= d log Zi

d log σ , (46) is an estimate of di, the dimension around xi, as a function of σ. Computationally, it is desirable to use the closed-form expression, for accuracy: 

Z′

> i

(σ) = 

∑|N ′(i)| 

> j=1

‖xi − xj ‖2 exp −‖ xi−xj ‖2

> 2σ2

σ2 ∑|N ′(i)| 

> j=1

exp −‖ xi−xj ‖2

> 2σ2

. (47) 54 3. A region of stability of Z′

> i

, i.e., a local maximum, is then an estimate of the dimension around 

xi.A local maximum (“peak”) in Z′

> i

(σ) can be interpreted as follows: as a ball around xi is expanded, the rate at which neighbors are seen has stopped increasing and must decrease with larger σ, since no additional neighbors can be found after the ball encompasses all points in N ′(i).Underlying is the assumption that N ′(i) is sufficiently representative of the manifold around xi.I.e., if neighbors are approximately uniformly distributed and dimensionality is constant within it, then Z′ 

> i

should remain constant over some appreciable range of σ, whence the notion of “stability”. Even though we work with a subset of X , there may still be multiple maxima in Z′

> i

, e.g., when the neighbors of xi are far from uniformly distributed around it. So, operationally, we use the global maximum of Z′

> i

, as this takes into account the information given by the majority of neighboring points. Now, because Zi → 1 as σ → 0, and Zi → N as σ → ∞ , the slope of log Zi

must approach 0 at both extremes, thus the global maximum of Z′ 

> i

must also be a relative one (a “peak”). We now proceed to avoid boundary effects by re-centering neighborhoods . The boundary, 

∂M, of a d-dimensional manifold (when present) has dimensionality d − 1 [68]. The correlation integral approach often fails for these—it typically returns d/ 2 for points in ∂M, since they have roughly half the number of neighbors compared to interior points. For the same reason, it tends to also underestimate d for points near the boundary. Since we work locally over a graph, we can regularize the computation by moving the focus to a more central, nearby point (thus regularizing over sampling artifacts as well): 4. Letting N (i) be the set of adjacent nodes to i in G? and including i itself, define ¯ι as the node 

j ∈ N (i) with smallest median squared distance to all points in the extended neighborhood 

N ′(i):

¯ι = argmin j∈N (i)median {‖xj − xl‖2), ∀l ∈ N ′(i)} . (48) Thus ¯ι is, in effect, the most central node in i’s immediate neighborhood 8.5. Use ¯ι as the point from which kernel values are computed for Zi(σ) by replacing xi with x¯ι in equation 45, thereby shifting the center of estimation of di. This assumes that the dimension does not change abruptly across neighboring points. Denote the resulting estimate by ˆdi.6. As with the MLE method (section 4.3), we may obtain a smoother estimate, ˆd′

> i

, by averaging over immediate neighbors in N (i):

ˆd′ 

> i

= 1

|N (i)|

∑ 

> j∈N (i)

ˆdj . (49) Finally, recall from section 3.5 that we also obtain a degree-based estimate, ˜di, when computing volume ratios (equation 36); we can use this information to further improve our results. A final estimate, d?i , is then obtained as follows:  

> 8Since we know G?, graph-theoretical quantities such as shortest-path betweenness centrality [45, 22] may also be used here.

55 7. To avoid overestimating the true dimension, compute an average ˜d′ 

> i

over N (i) as 

˜d′ 

> i

= 1

|N (i)|

∑ 

> j∈N (i)

⌊ ˜dj

⌋

= 1

|N (i)|

∑ 

> j∈N (i)

blog 2 deg( j)c . (50) 8. Compute the optimal estimate, d?i , as 

d?i = max 

{ ˆd′

> i

, ˜d′

> i

}

. (51) Application of this technique and comparison with other methods are given next. 

4.3.2 Experimental results 

Results of applying our neighborhood correlation dimension (NCD) algorithm compared to Lev-ina & Bickel’s MLE estimator (equation 41) are shown in Figures 32–34. For NCD, we compared results using IAN against those from k-NN graphs using various values of k (a range was chosen that included the best results for each algorithm). The IAN kernel was applied by using the dis-crete neighborhoods of G?, re-centered using neighbors-of-neighbors at most 3 hops away from i

(equation 48). Using IAN, we obtained near-optimal results for the stingray and the bent plane. For the 5-dimensional cylinder, the dimension was underestimated (mean 4.6). Methods based on correlation dimension are known to underestimate the true d when the sample size is not sufficiently large [25]. In these cases, the method of [26] can be applied a posteriori to improve results. For the MLE method, using large values of k tended to improve results, but only when di-mension was constant (as in the bent plane and cylinder datasets). For the stingray, however, no value of k gave correct results: small values of k increased the dimension estimates due to a bias, and large values tended to produce a uniform value throughout (thus giving better estimates only when d is constant). We found that computing the neighborhood averages using the correction of MacKay and Ghahramani [75], i.e., averaging the inverse of the estimators to reduce bias when k

is small, gave slightly better results. (We did not use the final smoothing procedure which involves choosing two additional neighborhood size parameters, k1 and k2.) Finally, we confirmed these observations by testing two additional datasets with non-uniform dimensionality (Figure 35). Again, while our algorithm achieved good results locally, there was no single value of k that allowed MLE to find appropriate local estimates everywhere. 

# 5 Summary and Conclusions 

In theory, applying the manifold assumption requires prior knowledge about the manifold: its geometry, topology, as well as how it was sampled. In practice, however, these manifold properties are rarely known. Instead, one typically imposes an assumption about the manifold’s dimension, d,which in turn suggests that k = 2 d nearest neighbors should suffice. This is how many—most!—of the data graphs underlying manifold inference and non-linear dimensionality reduction are built. Since it is difficult to know whether this assumption about dimension is accurate, it is common practice to test a few values of k and choose among the results. 56 NCD                  

> k= 2 k= 4 k= 8 IAN
> MLE
> k= 4 k= 8 k= 16 k= 32
> 1.0 1.5 2.0 2.5 3.0
> dimension

Figure 32: Estimation of local intrinsic dimension on the stingray dataset. Top row shows results for our neighborhood correlation dimension (NCD) algorithm using k-NN graphs with various 

k and using adaptive neighborhoods from G? (IAN). Bottom row shows results using Levina & Bickel’s MLE estimator, which was sensitive to the choice of k: using a small value grossly over-estimated the dimension over the body, and a large k ignored the geometry of the tail. NCD using IAN gave the best results, estimating dimension 2 for the body and 1 for the tail, with intermediate values for the transition tail-body and the boundary. Apart from the subjective nature of this choice, there are more general problems. Manifolds may not have a fixed dimension, they may be curved or with boundary, and sampling may vary. The intrinsic dimension may vary across the data, and so should the number of neighbors. In such cases, finding a compromise k may be far from ideal. We suggest a different approach: that one should build the nearest-neighbor graph, and hence the graph-Laplacian approximation, in as data-driven a manner as possible, while being aware of the manifold properties. Our algorithm of iterated adaptive neighborhoods (IAN) starts with a conservative assumption: that nearest neighbors should have no “nearer” neighbors between them. We then alternate between a discrete and a continuous view of neighborhood graphs, and use a volumetric statistic to check for outliers. A linear program keeps the scales minimal while providing a global cover. This optimization is convex, so results are deterministic; other approaches, such as t-SNE and UMAP, are stochastic, so depend critically on the initialization. Our kernel has been applied successfully to a variety of datasets, and compared against some of the most popular algorithms available. In all cases our performance dominates. Furthermore, IAN can be incorporated directly into many embedding algorithms, including diffusion maps, Isomap, UMAP, and t-SNE, improving their results. Most of these algorithms involve several free parame-ters; we have none other than the robust requirement for an outlier. 57 NCD                   

> k= 4 k= 16 k= 64 IAN
> MLE
> k= 8 k= 16 k= 32 k= 64
> 1.0 1.5 2.0 2.5 3.0 3.5
> dimension

Figure 33: Estimation of local intrinsic dimension on the bent plane dataset. As with the stingray (Figure 32), results are sensitive to the choice of k, but here a wider range of values work due to the constant dimension. For NCD, results with IAN are comparable to those using the best k-NN graph ( k = 16). With MLE, larger k improved results (comparable to those using NCD). 2 3 4 5local dimension counts NCD       

> k=8
> k=16
> k=32
> IAN
> 2345678

local dimension MLE 

> k=8 k=16 k=32

Figure 34: Estimation of local intrinsic dimension for the 5-D cylinder dataset of Figure 29. With NCD, results using IAN underestimated the true dimensionality (mean 4.63), but are still better than using a k-NN graph with arbitrary k. With MLE, larger values of k gave tighter distributions centered near the correct value (mean 4.86 for k = 32). Other popular embedding algorithms, e.g., LLE [88], approximate the tangent space over a 58 other-dims Figure 35: Dimensionality estimation for two datasets with non-uniform local di: a “tiara” (top row), where dimension varies smoothly from 2 to 1, and a “spinning top” (bottom row, middle cross-section shown), where dimension reduces from 3 to 1 as one moves from the bulky part toward the tip. Using the optimal k for MLE could not produce good results for the entire dataset (here, k = 32 for both datasets). The NCD method, in contrast, was able to correctly adapt to the local geometry by taking advantage of the data graph produced by the IAN kernel. local neighborhood around each point. Although not explored here, using G? to automatically pro-vide such neighborhoods is straightforward (analogous to what was done in section 4.3 to estimate the local dimensionality). Applications to clustering need to be explored. Our weighted graph has also been applied to geodesic estimation, achieving comparable results to those obtained from graph geodesics. In contrast, the graphs obtained from other similarity kernels produced less than optimal results. Our unweighted graph has found application in local dimensionality estimation. Our proposed algorithm, neighborhood correlation dimension (NCD), takes advantage of the adaptive connec-tivity of our graph to improve results based on correlation dimension, namely by restricting the correlation integral to an approximately tubular neighborhood around xi in M. As a result, we 59 obtained accurate estimates of the local dimension in datasets where it is not uniform. Several theoretical bounds are implied throughout this paper; these need to be proved. Multi-scale kernels, such as those from equations 6 and 7, are known to approximate Laplacian operators asymptotically [98, 16]. Using our application examples as evidence, we conjecture that our ver-sion also results in good approximations. In conclusion, understanding the interplay between manifold geometry, topology, and sampling lies at the heart of many data science applications. We have taken a first step to illustrate how discrete relates to continuous, how local estimates relate to global ones, and how uncertainties in data gathering relate to both. Applying data science in a way that leads to rigorous, scientifically-appropriate conclusions must take all of these into account. 

# A Greedy splitting 

As an alternative to the optimization from section 3.4 (which can be expensive when the number of edges in G is very large, mainly due to large dimensionality), we have developed a greedy approach in which scales that “ C-cover” each edge eij are assigned in decreasing order of length, 

rij (the Euclidean distance between xi and xj in Rn). We call this algorithm greedy splitting .Starting with the edge eij with largest rij , set σi = σj = Cr ij , with C ≤ 1, thereby satisfying 

σiσj ≥ (Cr ij )2 with equality—we say Cr ij is evenly “split” between σi and σj . Moreover, since 

rij = rFN  

> i

= rFN  

> j

, we know the constraints σi ≤ rFN  

> i

and σj ≤ rFN  

> j

are also satisfied. Continue with the edge eij that has the next largest length, rij . Here we are met with three possible cases in which a (re)assignment of scales is needed: 1. If neither of the nodes have been assigned a scale yet, evenly split the scaled distance between 

σi and σj , as above. 2. If one of the nodes does not have a scale yet (without loss of generality, let that node be j), set σ′ 

> j

to the minimum scale that ensures σiσ′ 

> j

≥ (Cr ij )2, i.e., σ′ 

> j

= ( Cr ij )2/σ i;3. If both nodes have previously been assigned a scale but eij is not C-covered by the current values of σi and σj , then set the quotient a = Cr ij  

> √σiσj

and update both scales: σ′ 

> i

= aσ i and 

σ′ 

> j

= aσ j , thereby evenly splitting the quotient between the two nodes. After cases (2) and (3), the updated scales might need to be “rebalanced” in order to meet the constraints σ′ 

> i

≤ rFN  

> i

and σ′ 

> j

≤ rFN  

> j

. Without loss of generality, let σ′ 

> i

> r FN  

> i

. Then, we set 

σ′′  

> i

= rFN  

> i

and σ′′  

> j

= σ′

> jσ′
> i
> σ′′
> i

. Only one of the two scales may exceed its upper bound: in (2), this is trivially true since only the newly-assigned scale may be greater than Cr ij ; in (3), since both σi

and σj have been previously assigned, we have σi ≤ rFN  

> i

and σj ≤ rFN  

> j

, as well as rij ≤ rFN  

> i

and 

rij ≤ rFN  

> j

, so therefore it must be the case that rFN  

> i

rFN  

> j

≥ r2 

> ij

= σ′

> i

σ′ 

> j

. Note that, as a corollary, both scales must meet their respective constraints after being re-balanced as above. The above is repeated until all edges have been visited. By covering the largest edges first, we assign the largest, most constrained scales first, allowing for the later, less constrained scales, to be as small as possible. Because in most cases this tends to evenly split the scaled edge lengths 

Cr ij between σi and σj , the algorithm produces reasonable (but usually sub-optimal) results when compared to the linear program of section 3.4.2. 60 References 

[1] E. Aamari, J. Kim, F. Chazal, B. Michel, A. Rinaldo, and L. Wasserman. Estimating the reach of a manifold. Electronic Journal of Statistics , 13(1):1359–1399, 2019. [2] N. Alon, S. Ben-David, N. Cesa-Bianchi, and D. Haussler. Scale-sensitive dimensions, uniform convergence, and learnability. Journal of the ACM , 44(4):615–631, 1997. [3] A. ´Alvarez-Meza, J. Valencia-Aguirre, G. Daza-Santacoloma, and G. Castellanos-Dom´ ınguez. Global and local choice of the number of nearest neighbors in locally linear embedding. Pattern Recognition Letters , 32(16):2171–2177, 2011. [4] N. Amenta and M. Bern. Surface reconstruction by Voronoi filtering. Discrete & Compu-tational Geometry , 22(4):481–504, 1999. [5] N. Amenta, M. Bern, and M. Kamvysselis. A new Voronoi-based surface reconstruction algorithm. In Proceedings of the 25th Annual Conference on Computer Graphics and Inter-active Techniques , SIGGRAPH, pages 415–421, 1998. [6] S. Arora and R. Kannan. Learning mixtures of separated nonspherical Gaussians. The Annals of Applied Probability , 15(1A):69–92, 2005. [7] S. Arora, W. Hu, and P. K. Kothari. An analysis of the t-SNE algorithm for data visualiza-tion. In Conference On Learning Theory , pages 1455–1462. PMLR, 2018. [8] R. Banisch, E. H. Thiede, and Z. Trstanova. pydiffmap. https://github.com/ DiffusionMapsAcademics/pyDiffMap , 2017. [9] E. Becht, L. McInnes, J. Healy, C.-A. Dutertre, I. W. H. Kwok, L. G. Ng, F. Ginhoux, and E. W. Newell. Dimensionality reduction for visualizing single-cell data using UMAP. 

Nature biotechnology , 37(1):38–44, 2019. [10] M. Belkin and P. Niyogi. Laplacian eigenmaps for dimensionality reduction and data repre-sentation. Neural Computation , 15(6):1373–1396, 2003. [11] M. Belkin and P. Niyogi. Semi-supervised learning on Riemannian manifolds. Machine learning , 56(1):209–239, 2004. [12] M. Belkin, J. Sun, and Y. Wang. Discrete Laplace operator on meshed surfaces. In Proceed-ings of the 24th Annual Symposium on Computational Geometry , pages 278–287, 2008. [13] M. Belkin, J. Sun, and Y. Wang. Constructing Laplace operator from point clouds in R d.In Proceedings of the 20th Annual ACM-SIAM Symposium on Discrete algorithms , pages 1031–1040. SIAM, 2009. [14] F. Bernardini, J. Mittleman, H. Rushmeier, C. Silva, and G. Taubin. The ball-pivoting algorithm for surface reconstruction. IEEE transactions on Visualization and Computer Graphics , 5(4):349–359, 1999. 61 [15] M. Bernstein, V. de Silva, J. C. Langford, and J. B. Tenenbaum. Graph approximations to geodesics on embedded manifolds (Technical Report), 2000. [16] T. Berry and J. Harlim. Variable bandwidth diffusion kernels. Applied and Computational Harmonic Analysis , 40(1):68–96, 2016. ISSN 1063-5203. [17] T. Berry, D. Giannakis, and J. Harlim. Nonparametric forecasting of low-dimensional dy-namical systems. Physical Review E , 91(3):032915, 2015. [18] A. Block, Z. Jia, Y. Polyanskiy, and A. Rakhlin. Intrinsic dimension estimation. Journal of Machine Learning Research , 22:1–30, 2021. [19] J.-D. Boissonnat, L. J. Guibas, and S. Y. Oudot. Manifold reconstruction in arbitrary dimen-sions using witness complexes. Discrete & Computational Geometry , 42(1):37–70, 2009. [20] J.-D. Boissonnat, A. Lieutier, and M. Wintraecken. The reach, metric distortion, geodesic convexity and the variation of tangent spaces. Journal of Applied and Computational Topol-ogy , 3(1):29–58, 2019. [21] S. P. Boyd and L. Vandenberghe. Convex optimization . Cambridge University Press, 2004. [22] U. Brandes. A faster algorithm for betweenness centrality. Journal of Mathematical Soci-ology , 25(2):163–177, 2001. [23] C. Bregler and S. Omohundro. Nonlinear image interpolation using manifold learning. 

Advances in Neural Information Processing Systems , 7, 1994. [24] F. Camastra. Data dimensionality estimation methods: a survey. Pattern Recognition , 36 (12):2945–2954, 2003. [25] F. Camastra and A. Staiano. Intrinsic dimension estimation: Advances and open problems. 

Information Sciences , 328:26–41, 2016. [26] F. Camastra and A. Vinciarelli. Estimating the intrinsic dimension of data with a fractal-based method. IEEE Transactions on Pattern Analysis and Machine Intelligence , 24(10): 1404–1407, 2002. [27] D. M. Chan, R. Rao, F. Huang, and J. F. Canny. t-SNE-CUDA: GPU-accelerated t-SNE and its applications to modern data. In 30th International Symposium on Computer Architecture and High Performance Computing , SBAC-PAD, pages 330–338. IEEE, 2018. [28] R. R. Coifman and S. Lafon. Diffusion maps. Applied and computational harmonic analy-sis , 21(1):5–30, 2006. [29] R. R. Coifman, S. Lafon, A. B. Lee, M. Maggioni, B. Nadler, F. Warner, and S. W. Zucker. Geometric diffusions as a tool for harmonic analysis and structure definition of data: Diffu-sion maps. Proceedings of the National Academy of Sciences , 102(21):7426–7431, 2005. 62 [30] R. R. Coifman, Y. Shkolnisky, F. J. Sigworth, and A. Singer. Graph Laplacian tomography from unknown random projections. IEEE Transactions on Image Processing , 17(10):1891– 1899, 2008. [31] J. A. Costa, A. Girotra, and A. O. Hero. Estimating local intrinsic dimension with k-nearest neighbor graphs. In IEEE/SP 13th Workshop on Statistical Signal Processing , pages 417– 422. IEEE, 2005. [32] K. Crane, C. Weischedel, and M. Wardetzky. Geodesics in heat: A new approach to com-puting distance based on heat flow. ACM Transactions on Graphics , 32(5):1–11, 2013. [33] S. Dasgupta. Learning mixtures of Gaussians. In Proceedings of the 40th Annual Symposium on Foundations of Computer Science , FOCS, pages 634–644, USA, 1999. IEEE Computer Society. ISBN 0769504094. [34] X. Desquesnes, A. Elmoataz, and O. L´ ezoray. Eikonal equation adaptation on weighted graphs: fast geometric diffusion process for local and non-local image and data processing. 

Journal of Mathematical Imaging and Vision , 46(2):238–257, 2013. [35] G. Dimitriadis, J. P. Neto, and A. R. Kampff. t-SNE visualization of large-scale neural recordings. Neural Computation , 30(7):1750–1774, 2018. [36] D. L. Donoho and C. Grimes. Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data. Proceedings of the National Academy of Sciences , 100(10):5591– 5596, 2003. [37] R. Dyer, H. Zhang, and T. M¨ oller. Gabriel meshes and Delaunay edge flips. In 2009 SIAM/ACM Joint Conference on Geometric and Physical Modeling , pages 295–300, 2009. [38] J.-P. Eckmann and D. Ruelle. Fundamental limitations for estimating dimensions and lya-punov exponents in dynamical systems. Physica D: Nonlinear Phenomena , 56(2-3):185– 187, 1992. [39] E. Facco, M. d’Errico, A. Rodriguez, and A. Laio. Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports , 7(1):1–8, 2017. [40] A. M. Farahmand, C. Szepesv´ ari, and J.-Y. Audibert. Manifold-adaptive dimension esti-mation. In Proceedings of the 24th International Conference on Machine Learning , pages 265–272, 2007. [41] H. Federer. Curvature measures. Transactions of the American Mathematical Society , 93 (3):418–491, 1959. [42] C. Fefferman, S. Mitter, and H. Narayanan. Testing the manifold hypothesis. Journal of the American Mathematical Society , 29(4):983–1049, 2016. [43] C. Fefferman, S. Ivanov, Y. Kurylev, M. Lassas, and H. Narayanan. Fitting a putative mani-fold to noisy data. In Proceedings of the 31st Conference On Learning Theory , volume 75, pages 688–720. PMLR, 2018. 63 [44] S. Fortune. Voronoi diagrams and Delaunay triangulations. Computing in Euclidean Geom-etry , pages 225–265, 1995. [45] L. C. Freeman. A set of measures of centrality based on betweenness. Sociometry , pages 35–41, 1977. [46] Y. Fujiwara, Y. Ida, S. Kanai, A. Kumagai, and N. Ueda. Fast similarity computation for t-SNE. In 2021 IEEE 37th International Conference on Data Engineering (ICDE) , pages 1691–1702. IEEE, 2021. [47] K. Fukunaga and D. R. Olsen. An algorithm for finding intrinsic dimensionality of data. 

IEEE Transactions on Computers , 100(2):176–183, 1971. [48] K. R. Gabriel and R. R. Sokal. A new statistical approach to geographic variation analysis. 

Systematic Zoology , 18(3):259–278, 1969. [49] C. Genovese, M. Perone-Pacifico, I. Verdinelli, and L. Wasserman. Minimax manifold estimation. Journal of Machine Learning Research , 13(43):1263–1291, 2012. [50] Y. Goldberg and Y. Ritov. Local procrustes for manifold embedding: a measure of embed-ding quality and embedding algorithms. Machine learning , 77(1):1–25, 2009. [51] P. Grassberger and I. Procaccia. Measuring the strangeness of strange attractors. In The theory of chaotic attractors , pages 170–189. Springer, 2004. [52] L. Haghverdi, F. Buettner, and F. J. Theis. Diffusion maps for high-dimensional single-cell analysis of differentiation data. Bioinformatics , 31(18):2989–2998, 2015. [53] G. Haro, G. Randall, and G. Sapiro. Translated Poisson mixture model for stratification learning. International Journal of Computer Vision , 80(3):358–374, 2008. [54] J. He, L. Ding, L. Jiang, Z. Li, and Q. Hu. Intrinsic dimensionality estimation based on manifold assumption. Journal of Visual Communication and Image Representation , 25(5): 740–747, 2014. [55] M. Hein and J.-Y. Audibert. Intrinsic dimensionality estimation of submanifolds in Rd.In Proceedings of the 22nd international conference on Machine learning , pages 289–296, 2005. [56] M. Hein and M. Maier. Manifold denoising. Advances in Neural Information Processing Systems , 19, 2006. [57] G. E. Hinton and S. Roweis. Stochastic neighbor embedding. Advances in Neural Informa-tion Processing Systems , 15, 2002. [58] B. K´ egl. Intrinsic dimension estimation using packing numbers. Advances in Neural Infor-mation Processing Systems , 15, 2002. [59] Y. Keller, R. R. Coifman, S. Lafon, and S. W. Zucker. Audio-visual group recognition using diffusion maps. IEEE Transactions on Signal Processing , 58(1):403–413, 2009. 64 [60] M. G. Kendall. Rank correlation methods . Griffin, 1948. [61] M. Kleindessner and U. Luxburg. Dimensionality estimation without distances. In Pro-ceedings of the eighteenth International Conference on Artificial Intelligence and Statistics ,volume 38, pages 471–479, San Diego, California, USA, 09–12 May 2015. PMLR. [62] W. R. Knight. A computer method for calculating kendall’s tau with ungrouped data. Jour-nal of the American Statistical Association , 61(314):436–439, 1966. [63] D. Kobak and P. Berens. The art of using t-SNE for single-cell transcriptomics. Nature communications , 10(1):1–14, 2019. [64] D. Kobak and G. C. Linderman. Initialization is critical for preserving global data structure in both t-SNE and UMAP. Nature biotechnology , 39(2):156–157, 2021. [65] O. Kouropteva, O. Okun, and M. Pietik¨ ainen. Selection of the optimal parameter value for the locally linear embedding algorithm. FSKD , 2:359–363, 2002. [66] S. Lafon. Diffusion maps and geometric harmonics . PhD thesis, Yale University, 2004. [67] S. Lafon, Y. Keller, and R. R. Coifman. Data fusion and multicue data matching by diffusion maps. IEEE Transactions on Pattern Analysis and Machine Intelligence , 28(11):1784–1797, 2006. [68] J. Lee. Introduction to topological manifolds , volume 202. Springer Science & Business Media, 2010. [69] J. A. Lee and M. Verleysen. Nonlinear dimensionality reduction , volume 1. Springer, 2007. [70] E. Levina and P. Bickel. Maximum likelihood estimation of intrinsic dimension. Advances in Neural Information Processing Systems , 17, 2004. [71] O. Lindenbaum, M. Salhov, A. Yeredor, and A. Averbuch. Gaussian bandwidth selection for manifold learning and classification. Data Mining and Knowledge Discovery , 34(6): 1676–1712, 2020. [72] G. C. Linderman, M. Rachh, J. G. Hoskins, S. Steinerberger, and Y. Kluger. Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. Nature Methods , 16(3):243–245, 2019. [73] A. V. Little, M. Maggioni, and L. Rosasco. Multiscale geometric methods for data sets I: multiscale SVD, noise and curvature. Applied and Computational Harmonic Analysis , 43 (3):504–567, 2017. [74] L. Lov´ asz. Discrete and continuous: two sides of the same? In Visions in Mathematics ,pages 359–382. Springer, 2010. [75] D. MacKay and Z. Ghahramani. Comments on ‘Maximum likelihood estimation of intrinsic dimension’ by E. Levina and P. Bickel (2005). The Inference Group Website, Cavendish Laboratory, Cambridge University , 2005. URL http://www.inference.org.uk/ mackay/dimension/ .65 [76] D. W. Matula and R. R. Sokal. Properties of Gabriel graphs relevant to geographic variation research and the clustering of points in the plane. Geographical Analysis , 12(3):205–222, 1980. [77] L. McInnes, J. Healy, N. Saul, and L. Großberger. UMAP: Uniform manifold approximation and projection. Journal of Open Source Software , 3(29):861, 2018. [78] N. Mekuz and J. K. Tsotsos. Parameterless Isomap with adaptive neighborhood selection. In Joint Pattern Recognition Symposium , pages 364–373. Springer, 2006. [79] G. Mishne and I. Cohen. Multiscale anomaly detection using diffusion maps. IEEE Journal of selected topics in signal processing , 7(1):111–123, 2012. [80] K. R. Moon, D. van Dijk, Z. Wang, S. Gigante, D. B. Burkhardt, W. S. Chen, K. Yim, A. v. d. Elzen, M. J. Hirn, R. R. Coifman, N. Ivanova, G. Wolf, and S. Krishnaswamy. Visualizing structure and transitions in high-dimensional biological data. Nature Biotechnology , 37(12): 1482–1492, 2019. [81] H. Narayanan and S. Mitter. Sample complexity of testing the manifold hypothesis. Ad-vances in Neural Information Processing Systems , 23, 2010. [82] P. Niyogi, S. Smale, and S. Weinberger. Finding the homology of submanifolds with high confidence from random samples. Discrete & Computational Geometry , 39(1):419–441, 2008. [83] P. Niyogi, S. Smale, and S. Weinberger. A topological view of unsupervised learning from noisy data. SIAM Journal on Computing , 40(3):646–663, 2011. [84] F. No´ e, R. Banisch, and C. Clementi. Commute maps: separating slowly mixing molecular configurations for kinetic modeling. Journal of Chemical Theory and Computation , 12(11): 5620–5630, 2016. [85] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research , 12:2825–2830, 2011. [86] K. W. Pettis, T. A. Bailey, A. K. Jain, and R. C. Dubes. An intrinsic dimensionality estima-tor from near-neighbor information. IEEE Transactions on Pattern Analysis and Machine Intelligence , 1(1):25–37, 1979. [87] S. Roweis, L. Saul, and G. E. Hinton. Global coordination of local linear models. Advances in Neural Information Processing Systems , 14, 2001. [88] S. T. Roweis and L. K. Saul. Nonlinear dimensionality reduction by locally linear embed-ding. Science , 290(5500):2323–2326, 2000. [89] A. Rozza, G. Lombardi, C. Ceruti, E. Casiraghi, and P. Campadelli. Novel high intrinsic dimensionality estimators. Machine learning , 89(1):37–65, 2012. 66 [90] M. Saerens, F. Fouss, L. Yen, and P. Dupont. The principal components analysis of a graph, and its relationships to spectral clustering. In European Conference on Machine Learning ,pages 371–383. Springer, 2004. [91] O. Samko, A. D. Marshall, and P. L. Rosin. Selection of the optimal parameter value for the Isomap algorithm. Pattern Recognition Letters , 27(9):968–979, 2006. [92] L. K. Saul, K. Q. Weinberger, F. Sha, J. Ham, and D. D. Lee. Spectral methods for dimen-sionality reduction. Semi-supervised Learning , 3, 2006. [93] B. Sch¨ olkopf and A. J. Smola. Learning with kernels: support vector machines, regulariza-tion, optimization, and beyond . MIT press, 2002. [94] D. Spielman. Spectral graph theory. Combinatorial Scientific Computing , 18, 2012. [95] J. Tang, J. Liu, M. Zhang, and Q. Mei. Visualizing large-scale and high-dimensional data. In Proceedings of the 25th International Conference on World Wide Web , pages 287–297, 2016. [96] J. B. Tenenbaum, V. de Silva, and J. C. Langford. A global geometric framework for non-linear dimensionality reduction. Science , 290(5500):2319–2323, 2000. [97] C. Th¨ ale. 50 years sets with positive reach—a survey. Surveys in Mathematics and its Applications , 3:123–165, 2008. [98] D. Ting, L. Huang, and M. I. Jordan. An analysis of the convergence of graph Laplacians. In Proceedings of the 27th International Conference on International Conference on Ma-chine Learning , ICML, pages 1079–1086, Madison, WI, USA, 2010. Omnipress. ISBN 9781605589077. [99] G. V. Trunk. Statistical estimation of the intrinsic dimensionality of a noisy signal collection. 

IEEE Transactions on Computers , 100(2):165–171, 1976. [100] L. van der Maaten. Accelerating t-SNE using tree-based algorithms. Journal of Machine Learning Research , 15(1):3221–3245, 2014. [101] L. van der Maaten and G. E. Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research , 9(11), 2008. [102] L. van der Maaten, E. Postma, and J. van den Herik. Dimensionality reduction: a compara-tive review. Journal of Machine Learning Research , 10(66-71):13, 2009. [103] S. Vempala and G. Wang. A spectral algorithm for learning mixture models. Journal of Computer and System Sciences , 68(4):841–860, 2004. [104] P. J. Verveer and R. P. W. Duin. An evaluation of intrinsic dimensionality estimators. IEEE Transactions on Pattern Analysis and Machine Intelligence , 17(1):81–86, 1995. 67 [105] X. Wan, W. Wang, J. Liu, and T. Tong. Estimating the sample mean and standard deviation from the sample size, median, range and/or interquartile range. BMC Medical Research Methodology , 14(1):1–13, 2014. [106] J. Wang, Z. Zhang, and H. Zha. Adaptive manifold learning. Advances in Neural Informa-tion Processing Systems , 17, 2004. [107] Y. Wang, H. Huang, C. Rudin, and Y. Shaposhnik. Understanding how dimension reduction tools work: An empirical approach to deciphering t-SNE, UMAP, TriMap, and PaCMAP for data visualization. Journal of Machine Learning Research , 22(201):1–73, 2021. [108] M. Wattenberg, F. Vi´ egas, and I. Johnson. How to use t-SNE effectively. Distill , 1(10):e2, 2016. [109] K. Q. Weinberger and L. K. Saul. Unsupervised learning of image manifolds by semidefinite programming. International Journal of Computer Vision , 70(1):77–90, 2006. [110] L.-S. Young. Dimension, entropy and lyapunov exponents. Ergodic theory and dynamical systems , 2(1):109–124, 1982. [111] L. Zelnik-Manor and P. Perona. Self-tuning spectral clustering. In Proceedings of the 17th International Conference on Neural Information Processing Systems , NIPS, pages 1601– 1608, Cambridge, MA, USA, 2004. MIT Press. [112] Z. Zhang and H. Zha. Principal manifolds and nonlinear dimensionality reduction via tan-gent space alignment. SIAM Journal on Scientific Computing , 26(1):313–338, 2004. 68
