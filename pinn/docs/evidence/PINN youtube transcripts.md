# Physics Informed Neural Networks (PINNs) [Physics Informed Machine Learning]

![Thumbnail](https://img.youtube.com/vi/-zrY7P2dVC4/maxresdefault.jpg)

👤 [Steve Brunton](https://www.youtube.com/channel/UCm5mt-A4w61lknZ9lCsZtBw)  🔗 [Watch video](https://www.youtube.com/watch?v=-zrY7P2dVC4&pp=ygUgcGh5c2ljcyBpbmZvcm1lZCBuZXVyYWwgbmV0d29ya3M%3D)

> welcome back I'm Steve Brunton from the
> University of Washington and today I'm
> really excited to start a new lecture
> series which is a part of a larger uh
> set of boot camps and short courses on
> the topic of physics informed machine
> learning so this is one of the most
> important topics in all of machine
> learning because it's really at the
> intersection of how do you leverage you
> know existing physical knowledge to
> improve machine learning systems and how
> do you use machine learning to learn new
> physics that humans have never written
> down before okay so um you know the idea
> of machine learning we talk about this
> all the time we see the success stories
> in the news in our daily lives and the
> pace of progress is really you know
> astounding in machine learning things
> that were once only you know uh science
> fiction are now science fact so this is
> you know the Mona Lisa coming to life
> from a single still image with machine
> learning we also have you know uh
> rapidly advancing generative AI so this
> is an example of Dolly 2 where you can
> put in a prompt like guitar monster and
> it just generates this beautiful image
> of a monster playing a guitar notice
> that the guitar is also a monster which
> I think is kind of cool and although
> these are absolutely useful for things
> like you know creating new art and
> graphics and generating text and transla
> languages all kinds of um you know
> classic machine learning tasks there is
> this growing world of how do we use
> physics inform machine learning for
> things like designing an entirely new
> aircraft or a new super material that's
> going to be used in the engine or in the
> the wing or the fuselage of this
> aircraft or any of the other um you know
> engineering and physics tasks that we as
> humans have things like you know
> predicting and modeling and
> understanding fluid uh flows and
> turbulence which are again important for
> Designing you know uh aircraft race cars
> transport ships as well as wind turbines
> and you know understanding climate
> change and weather and things like that
> things like Robotics and digital twins
> uh for
> autonomy uh and this is just a tiny tiny
> glimpse of the many many applications of
> machine learning and physics informed
> machine learning in particular
> in engineering and Natural Sciences so
> if you you know want to be a part of
> this uh Revolution if you want to change
> the world if you want to design new
> super materials that are going to allow
> us to build better aircraft and better
> wind turbines and um you know better
> algorithms for robots and understanding
> things like climate change physics and
> for machine learning is going to become
> an essential tool in Your Arsenal in the
> future so that's what this lecture
> series is about and this is the overview
> lecture uh in that
> series okay so zooming out we've talked
> about this a lot before that machine
> learning is essentially the process of
> building models from data using
> optimization and regression techniques
> now we've been building models from data
> with optimization and regression for
> decades if not centuries if not
> Millennia as humans so some of our
> earliest models of the planets were
> essentially models from observational
> data that we did some kind of a crude
> regression uh or optimization to kind of
> understand those
> models but today we have much more data
> it's much more we have much more
> advanced optimization algorithms and so
> that's one of the reasons machine
> learning has been advancing so
> rapidly now that's classic machine
> learning what we're interested in
> talking about today and in this series
> is physics informed machine learning
> which is how do we build physical models
> from data using those same optimization
> and regression techniques or at least
> building on those existing optimization
> uh and regression
> Frameworks and so this entire series is
> focused on this intersection of
> artificial intelligence and machine
> learning with physics okay I think too
> often researchers in the machine
> learning field kind of forget that we
> actually have uh centuries of knowledge
> of physics we have tons of tons of
> knowledge of how physical systems
> actually work and often times we throw
> that out when we build these machine
> learning models but in the last five or
> 10 years it started to become more clear
> that we actually need to incorporate
> physics into the machine learning
> process to get the advanced performance
> we need on really complicated systems
> like in engineering uh or in Natural
> Sciences and you know biology things
> like that and so there's kind of two big
> aspects that are going to going to come
> up over and over again two big themes in
> this series one is we can enforce
> physics into machine learning models we
> essentially can take known physics or
> partially known physics things like
> symmetries conservation laws invariant
> things like that and we can bake those
> in to our machine learning models to
> make them much more performant so we can
> learn better models that generalize
> better with less training data if we
> enforce physics in to the models and the
> flip side of that is we can often and
> increasingly are are able to use machine
> learning techniques to discover entirely
> new physics with our measurement data so
> if I have measurement data from you know
> galactic motion or a plasma or you know
> some biological system I might be able
> to use these machine learning methods uh
> that are being developed in other fields
> of computer science and I might be able
> to use those methods to discover
> new physical models that we've never
> been able to write down before that
> humans haven't been able to discover uh
> because of either their complexity or or
> some other aspect um you know some
> complexity of the data and so that's a
> really exciting aspect too we can start
> getting things like ordinary and partial
> differential equations that describe uh
> really really complex systems that
> humans haven't been able to describe
> using classical methods okay so those
> are the two big themes we're going to
> enforce or bake in physics into the
> machine learning process and
> simultaneously often times we're going
> to use machine learning to discover new
> physics and we're going to find that
> these are kind of dual uh
> problems and here's a really really
> simple example I like to show just to
> get us warmed up what do we mean by
> physics inform form machine learning
> this is a pendulum uh mounted on the
> wall in my lab and so it's just moving
> back and forth there's a little uh
> accelerometer on the tip and a little
> LED so you can take a video of this
> and kind of traditional machine learning
> might take a video of this pendulum and
> try to use an autoencoder neural network
> to uh reduce the dimension of this video
> into a minimal latent State given by
> this red Vector here such that you can
> use that minimal State and then recreate
> the video from those very few variables
> okay we know that this is a low
> dimensional system we think that this is
> probably described by you know a
> variable Theta the angle maybe it's
> velocity kind of naive machine learning
> would just you know try to find some
> coordinates that you can compress this
> data down to so that you could also then
> expand it back into the full video
> whereas physics inform machine learning
> is trying to do something a little bit
> more sophisticated we're not just trying
> to learn the coordinates uh to compress
> this video we're also trying to learn
> maybe a differential equation for how
> those coordinates evolve in time
> something like this equation we know
> from the the damped pendulum equation
> and again you can think of this in one
> of two ways I could either if I knew
> that this was the equation I could try
> to bake that into this latent State uh
> in these these red coordinates here to
> try to learn the right coordinates um
> with my machine learning model if I
> baked in this known physics I might be
> able to do a better job or alternatively
> I could try to learn this differential
> equation purely from this measurement
> data now again zooming into this example
> we know that there's kind of two key
> ingredients if we were going to do this
> using textbook methods the first would
> be I need to learn a good coordinate
> system Theta in which I can represent
> this uh pendulum motion Theta is a lot
> better than the X and Y position of the
> tip for example and then once I learn
> that coordinate I want to learn some
> equations of motion some differential
> equation that describes how the system
> evolves in time so that's kind of what
> we mean by physics and form machine
> learning often times that means learning
> the right uh coordinates and Lear
> learning some governing equations or
> some physical laws which often times are
> things like differential equations that
> describe the motion in time okay and
> again we can either learn that from our
> data or we can bake some of these
> elements into our machine learning
> models
> good okay so we're going to zoom out and
> we're going to be talking about
> specifically the stages that go into
> building a classic machine learning
> model so I'm going to break this down
> into five key stages that we typically
> use to build any machine learning model
> and then I'm going to show you how we
> can make physics into each of those five
> stages now this is not 100% accurate I I
> say you know this is 80 90% accurate
> tops but this is a good kind of
> guideline for how the process of machine
> Learning Works that we can use to kind
> of organize this this lecture Series so
> the first of five stages in machine
> learning is essentially deciding on a
> problem what are we going to model with
> our machine learning model okay so that
> means deciding on what are the inputs
> and the outputs what relationship am I
> trying to model um so it could be you
> know I have pictures of dogs and cats
> and I want to build a classifier to
> label them as dogs or cats we've decided
> on a problem of what we're going to
> model OKAY stage two once I've done that
> I need to gather and curate data
> training data that I'm going to use to
> train that model okay so what data will
> inform that model and how do I gather
> that data so again in that dogs and cats
> example I need a bunch of pictures of
> dogs and cats and I have to decide do I
> actually have the labels do I know which
> images are dogs and which ones are cats
> you know that that's my my data curation
> often one of the most expensive
> processes uh in this pipeline sometimes
> this can be you know millions or tens of
> millions of dollars to get that data
> set stage three and this is where a lot
> of people get excited uh myself included
> is stage three you get to design some
> kind of an architecture so we know what
> problem we're trying to model we know
> the inputs and the outputs uh that we
> want to be kind of learning the
> relationship between stage three is
> where we choose an architecture like a
> you know special type of neural network
> or a Cindy model or an auto encoder
> whatever type of architecture you choose
> that because you think that it's going
> to do a good job of representing that
> functional relationship between the
> inputs and outputs and so a lot of work
> goes into designing custom architectures
> that are good for specific types of
> problems we know that if we want to do
> image classification we use
> convolutional neural networks if we want
> to model time series we use recurrent
> neural networks and things like that so
> there's a lot of kind of knowhow in
> choosing the architecture in fact this
> is an architecture of an auto encoder
> that is used for compressing the
> dimension of an input down before
> expanding it back out okay stage four so
> now we know the problem we have our data
> and we have a candidate architecture we
> think we might be able to train to model
> that input output relationship stage
> four is crafting a loss function some
> kind of an objective function that tells
> you if your models did a good job so
> usually this includes terms like the
> actual error of the input output um
> prediction of your neural network or
> whatever your machine learning
> architecture is with the actual data you
> use to train it and there's all kinds of
> other loss functions you can add you can
> add regularization terms to make your
> model more sparse or more smooth or
> things like that so again uh steps three
> and four are where a lot of the modern
> research uh in machine learning and in
> physics machine informed machine
> learning uh exists today okay a lot a
> lot of research in stages three and
> four and step five once we have our
> architecture and our loss function we
> essentially pick some optimization
> algorithm to train the model okay and by
> training the model what we mean is we
> use optimization to tweak the parameters
> of our architecture to tune and learn
> the parameters of this architecture to
> minimize our loss function averaged over
> our data this is such an important point
> I'm going to say this again okay so our
> architecture is some class of functions
> that you can use to represent the input
> output relationships you want to to
> model okay so there are essentially
> parameters of this architecture you can
> tune in the case of a neural network
> it's literally the weights of all of the
> connections in that neural network you
> get to tune those weights to learn some
> kind of input output function and so we
> use optimization like the atom Optimizer
> and many other types of you know
> stochastic gradient descent
> optimization
> to to tune those parameters of your
> architecture to minimize your loss
> function averaged over your training
> data and that's really the entire
> process of machine learning okay we
> decide on a problem we collect our data
> we pick a class of functions we think
> can represent that input output
> relationship we call that an
> architecture like a specific kind of
> neural network we design or craft a loss
> function that tells us if we did a good
> job and then we use optimization to
> tweak the parameters of our architecture
> to learn those parameters that minimize
> the loss function averaged over our
> training
> data and the reason I lay it out like
> this in these five stages is because
> each of these five stages gives unique
> opportunities for embedding physics into
> the process and in some cases for
> discovering physics so if we have
> partial knowledge of the physics if I
> know that I am trying to model you know
> the lift over an air foil or some
> material property or some robotic arm
> with you know joints and linkages then
> at each of these stages each of these
> five stages I can embed knowledge of the
> physics and improve this machine
> learning process I can constrain the
> architecture I can add custom loss
> functions there are certain types of
> optimizations I can use that will
> respect this physics more than others
> and so all of these stages give us
> unique and Powerful ways of adding
> physics into the
> process so we're going to use this
> diagram to organize the almost this
> entire kind of uh short you know module
> or boot camp in this larger lecture
> Series so we're going to zoom into each
> of these five examples and we're going
> to look at how uh how to embed physics
> and extract physics in each of these
> stages of machine
> learning so um if you want to learn more
> about this uh some of this is in a
> textbook written by Nathan Kutz and
> myself datadriven science and
> engineering you can download a free pdf
> uh of it here and I'm hoping that you
> know around the time that this video is
> released there will be other resources
> you know notes on this and things like
> that uh so check the um video
> description for more links to resources
> um that that you can
> use now again we think a lot about
> architectures that's one of the kind of
> exciting things that gets people you
> know a lot of people get excited by all
> of the cool architectures you can try on
> a new uh machine learning task
> especially in physics inform formed
> machine
> learning and there is a zoo of possible
> you know architectures just for neural
> networks and I want to make an important
> Point not all machine learning is just
> neural networks there's a lot of other
> architectures that are not neural
> networks that we can use um to learn
> models from data but even within the
> realm of just looking at neural networks
> there is this kind of managerie this
> this zoo of neural networks and a lot of
> times what we do is kind of akin to uh
> Alchemy so kind of modern day Al Alchemy
> where you have a problem and through
> some combination of intuition maybe
> reading Reddit posts or seeing what your
> friend did we have some idea of what
> types of neural network architectures
> might or might not work for our problem
> and then we often end up just trying a
> bunch of them and sometimes they work
> and sometimes they don't sometimes you
> can turn lead into gold metaphorically
> speaking and sometimes you can't and so
> one of the big pushes of applying
> machine learning to physical systems
> where we actually do have some prior
> knowledge we know a lot about these
> systems is that maybe we can start also
> learning principles of how and when
> different neural network architectures
> actually are appropriate for a given
> problem so this is a really really
> important point you know obviously we
> want to be doing better engineering and
> better science and we want to use the
> best tools at our disposal which
> nowadays you know machine learning is in
> that tool set but there is another
> perspective Ive which is by applying
> machine learning to physical problems
> where we sometimes know the answer we
> can learn a lot more about these um
> machine learning algorithms and models
> then we would if we just apply it to a
> system where we don't know the answer
> things like you know trying to cure
> cancer okay so if you apply this to
> systems where you know the answer we
> might be able to go from this Alchemy
> stage of machine learning to something
> that more closely resembles chemistry
> where there's actual principles and
> organization to when and how to use
> these building blocks for a particular
> problem okay um that's something we are
> very very you know interested in doing
> as a community there's a ton of effort
> uh kind of trying to learn these basic
> principles of machine learning by
> applying them to physics systems systems
> where we know the answer and Engineering
> Systems and this is actually one of the
> main thrusts of this NSF funded AI
> Institute in Dynamic
> systems um you can kind of go to the
> website and Google it or and check it
> out but there is this NSF funded AI
> Institute in Dynamic systems which is
> really focused on trying to understand
> how can you embed physics into machine
> Learning Systems how can you use machine
> Learning Systems to learn new physics to
> solve engineering tasks in these you
> know various application domains and to
> go from that kind of alchemy uh model of
> just you know guess and check and trial
> and error to something much more
> principled okay and so this is a
> collaboration with many many people and
> it's actually bigger than just this NSF
> Institute um trying to learn these
> principles of machine learning okay so
> really exciting uh field and topic and
> area and it's a really great time to be
> a researcher uh stepping into this field
> because it's you know there's a ton of
> open problems unsolved challenges and
> you know many many many uh billion dooll
> Industries can be transformed with these
> techniques uh in the coming
> years good okay so uh I'm just going to
> zoom into these five uh kind of areas of
> machine learning and talk about examples
> of how we bake physics in um this is
> just going to be a very very fast
> highlevel overview I'm going to have
> entire videos going into each of these
> sub problems so we're going to have a
> whole video talking about how you can
> bake physics into this stage one and
> stage two and so on and so forth but
> let's just kind of zoom in at a very
> high level so the first stage of machine
> learning is deciding on the problem
> sometimes this doesn't even go uh it
> goes unstated you know people take this
> for granted but this is an important
> part of the machine learning process is
> picking what you actually think you can
> model with your data what you want to
> model um what would be a useful model
> why do you want to model those are all
> important questions and fundamentally at
> the end of the day if you're modeling a
> physical system like you know a lava
> lamp or a pendulum or a fluid flow if
> you're modeling um you know a materials
> process or or something that is actually
> physical then you're kind of already
> doing this this you know physics based
> machine learning it physics is already
> in the process when you decide that
> you're trying to model some physical
> process so for example if I'm going to
> you know in my machine learning model
> try to learn the forces on this pendulum
> you know so we know f equals ma if I try
> to learn those forces with a machine
> learning model I'm already baking
> physics into the process through the
> problem statement similarly I could try
> to learn you know a hamiltonian or a
> lran I could try to learn uh a free
> energy potential or um you know lots and
> lots of different ways you can you can
> bake in physics just by setting up the
> problem in a clever
> way okay stage two curating the data so
> what data am I going to use again if the
> data comes from a physical system
> inherently you are kind of of embedding
> physics through um that that data
> collection and sometimes I joke that
> stage two is kind of the Google way of
> baking physics into the model because if
> you the idea is if you collect enough
> data of the natural world it kind of has
> to learn physics like FAL Ma and maybe
> even eventually things like eal mc^2
> just to reconcile all of that data that
> it's collecting but the reason I say
> that that's like the Google approach is
> because it's very very expensive that
> relies on having you know nearly
> infinite amounts of data which often as
> scientists and Engineers we don't have
> that luxury so we much much more often
> have targeted limited kind of narrow
> data sets that we have to use to learn
> the physics that extrapolates more
> broadly okay um and this is kind of
> subtle again we're going to have a whole
> video going into this but I want to to
> briefly mention it so you know for
> example am I going to take fluid flow
> Fields as my training data and I'm I am
> I going to have them at different flow
> velocities you know that's a very
> different model if it only has one fluid
> flow velocity or it has a range of
> velocities that will change kind of what
> physics I'm capturing in my model
> similarly you know if I'd have uh fluid
> flow geophysical fluid flow this is
> actually cloud formation uh past
> Guadalupe Island if I have this kind of
> data one of the things I can do in my
> data curation process to uh add physics
> is if I think that the physics doesn't
> actually depend on you know the angle of
> rotation or translation if I think that
> my physics is invariant to some kind of
> symmetry like translation or rotation I
> can augment my data with additional
> copies that are transformed according to
> that in this case you know rotations in
> other cases translations that's again a
> very very common thing people do is that
> if you know that your system has some
> symmetry or invariance which is another
> form of physics you can augment that
> data to kind of have uh to transform
> your data to have those symmetries and
> invariance so for example if I'm
> building a classifier that can tell you
> know a Prius from a Ford pickup truck
> and maybe you know other cars and trucks
> I'm going to take my images of cars and
> trucks and I might rotate them and
> translate them and scale them because
> none of that should matter for the
> classification so I can take my data and
> augment it it to include those
> symmetries uh and invariances that we
> think our model should be uh invariant
> to good um and again one of the things
> I'm going to talk about a lot and I've
> talked about this in previous videos and
> we're going to zoom into it here is the
> coordinates typically matter a lot when
> we're doing uh machine learning on
> physical systems so this is a cartoon uh
> from M and christon that I really like I
> use this one a lot that shows these kind
> of two different coordinate systems
> systems for the planets and the Sun and
> the Earth the one on the right this is
> the geocentric view that is kind of what
> things look like from our Earth centered
> perspective and it's really hard to
> learn physics and models in this
> coordinate system whereas if we fix the
> coordinate system and put the Sun at the
> center it's a lot easier the data makes
> more sense and it's easier to build a
> model from data so often curating the
> data means finding the right coordinate
> system so sometimes we learn that
> sometimes we kind of force that through
> prior knowledge of the physics and it
> makes a huge huge difference in the
> learning process good uh okay stage
> three designing an architecture we're
> going to spend a ton of time here
> probably you know five or 10 hours are
> going to be dedicated just to all of the
> different architectures on the market
> different loss functions for those
> architectures um again at a mile high we
> might have things like lran neural
> networks so if you know that your system
> has this kind of Oiler lrange equation
> uh framing if it's a mechanical system
> like a double pendulum you might use a
> specific architecture like a lran neural
> network um so that your system conserves
> energy important Point almost all of
> these architectures also have a
> coresponding set of custom loss
> functions that are needed to train them
> so stages three and four really do
> sometimes kind of uh get a little bit
> mixed okay so this is just one example
> of a particular structure for a physical
> system
> um we also have um one notion of of kind
> of what is physical is uh parsimony we
> like our models to be as simple as
> possible to describe the data but no
> simpler that's the approach here in our
> Cindy autoencoder where you use an
> autoencoder to learn a good coordinate
> system and then you have the sparest
> model in some class of models that
> describes the data okay so we're going
> to use that principle of low dimensional
> and sparse to promote models that are as
> simple as possible to describe the data
> and no simpler this principle of
> parsimony that's been with us for 2,000
> years from you know Aristotle to
> Einstein that's been one of the ways
> we've we've kind of measured if a system
> is physical is in its Simplicity and its
> ability to describe measurement data
> okay so this is another type of
> architecture um and then you know this
> is actually one of my favorite
> architectures it's it's one of the
> earlier ones in the modern physics
> inform formed machine learning era
> um by Julia Ling and
> collaborators they were using they were
> using neural networks to try to predict
> uh to try to build closure models for
> for turbulent fluid flows and their
> architecture is different than a
> standard deep neural network
> architecture they have this additional
> auxiliary kind of tenser input layer
> that allows their Network to be
> invariant to Galilean Transformations
> again we know that these fluid flows
> should be the same you know under
> certain kinds of rotations and
> translations and so through a choice of
> architecture they made it so that their
> models have to have that kind of uh
> invariance and they perform way way
> better than if you don't include that
> into the machine learning process these
> are just three of like literally
> hundreds of different types of custom
> architectures that you can use to imbue
> certain types of physics into your
> system now um I was talking to my wife
> Bing about this earlier and she pointed
> out when you choose an
> architecture to be to make your system
> more physical to make your machine
> learning model more physical that's
> something called an inductive bias it's
> kind of a bias that goes uh unspoken but
> it highly steers or influences the
> downstream machine learning process so
> we're going to talk about inductive
> biases uh and choices of architectures a
> bit more later in this section
> okay good uh moving on so you know item
> four this is another really really rich
> area of research is how do you build
> loss functions that promote models that
> are more physical in certain ways um we
> already talked about those sparse models
> parsimonious models that are as simple
> as possible to describe the data that is
> Quantified by a term in the loss
> function the L1 Norm or the l0 norm
> quantifies sparsity and parsimony so
> often times you know physics is promoted
> through additional regularizing terms or
> loss functions um in your training
> process probably the most famous example
> of this is the physics informed neural
> network or
> pin um Illustrated here so this is just
> a standard uh kind of you know deep feed
> forward neural network used to map some
> inputs like space and time to some
> outputs like a fluid flow field and
> normally what I would do is I would just
> have some loss function that averages
> you know how accurate is this model over
> some training data that's kind of the
> naive way of doing it what the pins does
> is that from this output data you can
> actually compute terms in your partial
> differential equation and you can add
> your actual physics the partial
> differential equation that the system
> should be satisfying as another loss
> term in your loss function so if you
> know the physics if you know that your
> system is Divergence free or satisfies
> The navier Stokes equations or the
> elastic beam equation whatever the
> equation is if you know it you can add
> in a regularizing loss term to your
> machine learning process and this should
> dramatically improve the learning
> performance you can get away with way
> less data and get much better model
> performance if you include the physics
> as a loss function so again we're going
> to have a whole lecture just on this one
> topic um because it's so important codes
> and examples and case studies and
> everything okay the the last step um and
> this one I think is super important it's
> not quite as common as stages three and
> four for embedding physics but it's very
> very important and we do a lot of work
> in this um in my lab and in my
> collaborators groups is using
> optimization to enforce physics so if I
> add physics as a term in the loss
> function that is essentially adding the
> suggestion that we want physics to be
> satisfied but it's always fighting you
> know we want low model
> and we want the physics to be satisfied
> and they sometimes are dueling
> objectives so just adding physics as a
> loss function only kind of promotes
> physics but it doesn't enforce that
> physics is satisfied but often with
> optimization you can constrain um and
> enforce that your physics is satisfied
> at a much higher Precision you know at a
> much um a much more stringent type of
> enforcing of physics so um for example
> we've done some work in in fluid flow
> modeling with machine learning and we
> know that there are certain symmetries
> and properties like energy is conserved
> in these incompressible flows and so
> what I could do is I could build a loss
> function that has my model error and
> some constraints that are necessary for
> this to conserve
> energy or instead of just having this as
> a loss function I could do a constrained
> lease squares so that these constraints
> are satisfied always to numerical
> Precision by Construction so again if
> this doesn't make perfect sense right
> now it's okay we're going to go into
> depth in each of these topics and have
> you know at least one video but probably
> a couple of hours of material on each of
> these topics but the idea is you can
> promote if I know some constraints that
> should be satisfied for my system to be
> physical things like energy conservation
> or a certain type of symmetry I can
> either add that in as a term in my loss
> function but it's not going to go to
> zero through the optimization process if
> it's just in my loss
> function or I can do things like use
> lrange multipliers or constrained lease
> squares so that I can change my
> optimization problem so that these
> constraints are exactly satisfied uh
> every you know every time I optimize
> this function so that's something we do
> a lot um and there are really powerful
> methods for enforcing physics this way
> it's a little more involved it gives you
> better physics imp position but it's
> also you know more human effort to
> actually build this into the optim ation
> function another example um so we're
> going to find kind of a theme in this
> lecture series is um we're going to talk
> a lot about what is physics in the first
> place what is this physics we're talking
> about and often times physics manifests
> itself through symmetries okay um I am
> bilaterally symmetric meaning I have
> mirror symmetry so if I flip myself in
> the mirror essentially nothing changes
> and lots of physical systems have other
> types of symmetries we've talked about
> translational symmetry rotational
> symmetry um this is you know a Quantum
> potential well with um a hexagonal
> potential so it has kind of this um
> interesting symmetry group of uh
> rotations and
> Reflections and um I I'll talk about
> this later but there are methods like
> physics inform DMD where you actually
> constrain your solution to live on a
> certain manifold of solutions
> satisfy that symmetry so this is again
> going to be a whole probably mini short
> course all about symmetries and how to
> bake symmetries into machine learning
> and how to extract symmetries uh from
> your data with machine learning huge
> huge topic very very important and this
> is going to change how we do machine
> learning it is currently changing how we
> do machine learning and it's only going
> to um continue to improve our
> capabilities in the next 5 10
> years okay so those are the five stages
> those are at least you know five clear
> opportunities of how to bake physics
> into your machine learning process now
> I'm just going to give you a super fast
> sneak peek of the rest of what you're
> going to see um at least in the next few
> hours um so you know I guess first what
> I'm going to do is I'm just going to
> show you kind of a pictoral example of
> those five stages just so you can see it
> in a different way and then I'm going to
> give you a sneak peek uh of some of the
> things we're going to do you know coming
> up next so again stage one is setting up
> the problem so I'm going to give I'm
> going to walk through a specific example
> now where the problem I want to solve is
> model reduction of a fluid flow so
> problem is I want to take a complicated
> model of a fluid and I want a simpler
> model of that fluid something that runs
> faster on a smaller
> computer so I need some data I need to
> curate some uh measurement data in this
> case maybe I have a simulation an
> expensive simulation of a full
> complexity
> you know numerical simulation of a fluid
> flow so my problem is trying to get a
> simpler description I start with the
> high-dimensional data stage three is
> choosing an
> architecture so in this case the
> architecture I'm going to choose is this
> kind of autoencoder architecture because
> it has the properties I want it goes
> from high dimension to low Dimension and
> these low dimensional States Z are
> chosen so that they are most likely to
> be able to reconstruct the high
> dimensional State again
> and I'm also going to say that I want to
> be able to learn a dynamical system some
> differential equation on that low
> dimensional latent State that's what I
> mean by model reduction I want to take
> my high dimensional system compress it
> down to a low dimensional State and
> learn some Dynamics some differential
> equation in that low dimensional State
> it'll be much much faster then to run
> this simulation maybe in real time and
> I'll get most of the physics that's
> happening in this system okay so those
> are the first three stages
> stage four is crafting a loss function
> so we need to build some kind of a
> function that tells me if my model did a
> good job now remember our architecture
> is some function that Maps inputs to
> outputs to do you know the problem to
> solve the problem we want to solve and
> it has parameters Theta these parameters
> are tuning parameters that I can tweak
> to make this function fit the data okay
> in this case I have you know my three
> neural networks each of them has a set
> of parameters Theta 1 Theta 2 and theta
> 3 and I can tweak those parameters to
> make this function fit this data better
> okay and that's Quantified by this loss
> function what I mean by by a better fit
> and finally the fifth stage is to
> optimize over all of those thetas
> literally to tweak the thetas to
> minimize the loss function averaged over
> the training data this is just a
> restatement of what we've already said
> but in a slightly different pictoral uh
> fashion
> okay okay good so again now we're going
> to zoom out do a sneak preview of what
> we're going to see next uh in this
> series so applications and Engineering I
> think this is a huge one we're going to
> have lots of case studies and
> applications looking into how this has
> actually been used in various use cases
> and what I'm going to show here is just
> a small window of the many many
> applications um that there are and we're
> going to explore in this series so
> things like again modeling fluid flows
> and
> turbulence um things like shape
> optimization how do you actually design
> a wing that has the right lift overd
> drag characteristics and that has the
> right structural characteristics this is
> a big multi-objective optimization
> problem and remember training a machine
> learning model is also a multi-objective
> optimization problem so these kind of
> make sense that they would go together
> so things like modeling fluid flows
> maybe designing you know the shape of an
> aircraft or a car or a wind
> turbine um designing a material a
> composite or an alloy that you're going
> to use in these engineering Downstream
> applications again multiobjective
> optimization very high-dimensional
> design space for uh for
> materials um digital twins so I might
> have some physical aspect like a robot
> arm or like an aircraft or a factory
> floor and I might want some digital
> representation some Digital model that I
> can design and optimize over uh like
> this digital twin here um and then kind
> of just robotics in general Robotics and
> autonomy is an area that is rapidly uh
> advancing with new machine learning
> technology and again this is just a
> pretty small um you know subset of the
> applications that there are and that
> we're going to talk about other examples
> include drug Discovery protein
> design um you know and those are systems
> that are governed by by physics there's
> physics and how proteins fold and how
> you know chemistry Works um so you know
> those are also big big applications um
> other areas things like climate science
> understanding weather phenomena and
> patterns and climate Trends again
> related to turbulence but a very very
> different context okay tons of
> applications uh of
> this um digital twin is really an
> important concept here when we think
> about building a machine learning model
> that is trained from data and has this
> kind of hybrid physics and data driven
> flavor that's very much what a digital
> twin is going for it's kind of a living
> uh model of some complex system like an
> aircraft or a factory floor or you know
> a materials uh baking process like a
> composite baking process where you want
> this digital uh twin of your physical
> asset to be as accurate as possible
> possible you want your model to update
> as you collect new data so your machine
> learning models need to be able to
> update you need to have uncertainty
> estimates of how good that model is or
> where it's the most uncertain so you can
> either know if you can trust your model
> and in you know the best case scenario
> you know how to go collect more data to
> improve your model all of those are
> working towards this notion of a digital
> twin which in principle should allow us
> to design to do better engineering
> design cheaper and faster and safer so
> we're gonna have again a whole short
> course on digital twin engineering where
> you know we talk about how do you build
> these digital twin models and how do you
> use them for better design uh testing
> evaluation things like that okay I
> really like this diagram here from this
> uh paper by uh captin at all in uh
> nature computational science and get
> kind of is a nice representation of this
> kind of Duality between the physical and
> the digital twin of that physical asset
> okay so um again physics and for machine
> learning if you're going to use a
> digital twin for something like an
> aircraft it better be you know physical
> it better have that physics baked into
> that
> process okay we're also going to talk
> about the importance of Benchmark
> systems so uh most of the progress in
> classic machine learning image Sciences
> natural language processing things like
> that a lot of that progress is because
> we had very good Benchmark problems and
> data sets that you could use that the
> community could use to test their
> methods on so a researcher didn't have
> to create a benchmark problem and the
> Machine learning algorithm they could
> focus their work on the machine learning
> algorithms okay and so increasingly
> we're finding that we need those kinds
> of Benchmark problems for Dynamics and
> control for physics and Engineering
> problems so I might need you know these
> static data sets of of our systems um
> they're not static in the sense the
> system is evolving in time but once I
> collect that data the data doesn't
> change it's a static data set of a
> dynamical system it could be simulated
> or
> experimental um but really I want to be
> you know developing these Benchmark
> systems
> for engineering purposes where we're
> actually trying to manipulate the
> behavior of the system we actually want
> to control the fluid with our machine
> learning model so I need to be able to
> interact with that simulation and
> actually close the loop with my machine
> learning model so we're developing
> Benchmark problems in that kind of
> interactive uh control framework as well
> and then ideally we would go you know to
> actual living cyber physical systems
> like an actual piece of laboratory
> equipment that you can train your
> machine learning model on in real time
> that's much closer to the kind of
> digital twin notion that we're going to
> be getting towards again this is going
> to be an entire short course on all of
> the benchmarks that are out there what
> are the characteristics that you need
> for a physics and for machine learning
> Benchmark and so on so this is uh an
> important topic we're going to come back
> to but we need to be thinking about you
> know how are we going to test our models
> and what is the ground truth do we have
> ground truth in some of these
> cases um architectures we already talked
> about the importance of that stage three
> and really it's architectures and the
> loss functions we use to train these
> architectures but we're going to cover a
> lot of the really really important ones
> things like uh res Nets and UNS we're
> going to cover uh fora neural operators
> and general kind of operator methods uh
> we'll talk about Cindy and you know kind
> of Library regression and parsimonious
> symbolic regression modeling we'll talk
> about physics and form neural networks
> uh deep operator networks and a bunch
> more this is just kind of a a quick
> smattering of some of the architectures
> we're going to look at but we're going
> to spend a lot of our time doing whole
> videos or even whole video series diving
> into these really really important
> topics how do you build these loss
> functions how do you train these what
> applications do they work on you know
> what how do we actually code this up and
> try it ourselves okay so we're going to
> really dig into a lot of these
> applications uh and you can actually
> take these then and use these to build
> your own um you know examples and and
> try this out
> yourself and again um we're going to
> throughout this entire series revisit
> this notion of like what is physics in
> the first Place why do we want to bake
> physics into our systems why do we need
> to bake physics into our systems when we
> try to learn physics from data what does
> that even mean what are we learning and
> time and time again we're going to find
> that things like conservation laws
> invariances and symmetries are some of
> the ways that we end up uh encapsulating
> physics so we're going to again see
> symmetries a lot you know in uh drug
> Discovery and protein folding there are
> symmetries fluid flows and mechanical
> systems uh material systems Quantum
> systems symmetries are fundamental to
> physics and so symmetries and
> conservation laws and invariances are
> going to keep coming up over and over
> again also this notion of parsimony or
> uh Simplicity of the model and in the
> process of doing this we're also going
> to not lose track of the history uh of
> Science and the history of physics
> because that's going to give us a lot of
> important parallels that we can learn
> from in this modern machine learning era
> so the difference between astrology and
> astronomy what changed the difference
> between Alchemy and chemistry okay and
> it really there is a lot of that uh
> parallel happening today in machine
> learning and so we can learn a lot from
> the history of science uh when we're
> doing this kind of new physics and form
> machine
> learning okay uh so that's the that's
> the topic that's the course um physics
> inform form machine learning we're going
> to have tons of short uh focused modules
> on each of these
> topics um I see this fitting into you
> know a larger coursework um that we're
> hoping you know to actually offer for
> credit uh at University of Washington at
> some point in the future so you know I
> hope you enjoy this it's a passion of
> mine it's one of the most important
> fields of machine learning and this is
> going to increasingly become uh you know
> very powerful tool in Your Arsenal to
> solve real world problems all right
> thank you you
> 
## Summary

The video provides a comprehensive introduction to Physics-Informed Neural Networks (PINNs), a powerful approach that combines traditional neural networks with physics constraints encoded in the loss function. The key innovation of PINNs is using automatic differentiation to compute derivatives needed for physical laws, and adding these as soft constraints during training.

> "it's based on a kind of neural network idea that also incorporates known physics in the form of a partial differential equation in the loss function"

> "what these authors did in this kind of classic now PINN paper is they extended this naive neural network picture... by computing things like partial derivatives of these output variables in terms of the input spatial and temporal variables... you can use that same idea in a modern language like PyTorch or JAX to compute these partial derivatives of the outputs with respect to the inputs"




> >> this paper um out of Michael Mahoney's uh kind of group and and collaboration on Co characterizing the actual failure modes in pins and I'm not going to expect you to read all of this you should download the paper and check it out because they have open source code and you can play with it yourself   
>    ... a set of physics pde problems where they can show under certain parameter regimes the method fails to train um and then essentially they looked at you know adding how what if you balance that that pte loss term so you can crank that pte loss term up or down that's a hyperparameter of how important ...
>    
>    28:54
>    
>    ... increase that physics loss from from zero to you know some large value that's a very clever and kind of straightforward idea and then uh their second idea is posing the learning problem as a sequence to sequence learning task so two concrete ideas for how to actually uh fix these failure modes and all ...
## Key points

- PINNs combine neural networks with physics-based loss terms
  - > "it takes a standard kind of neural network representation of the field variables you want to predict... and it computes the partial derivatives that are relevant for the physics in this problem... from those computed partial derivatives you can create this additional loss term that tells you if this known physics is being violated or not"

- PINNs can work effectively with limited data by leveraging physics constraints
  - > "this works really well for systems where we know something about the physics... even if I only have pretty limited data here I can still test if my network is physical... I can evaluate that on test points that are not actually in my training data"

- Physics constraints are only enforced softly as part of the loss function
  - > "because the physics is added as a loss function that's a strength because it's very intuitive and easy but it also only suggests that your physics is being satisfied... you're almost never going to actually have this purple term here be exactly zero"

- PINNs excel at reconstruction from sparse measurements
  - > "it's really really good for estimating things like whole flow fields with fairly sparse sensor measurements... taking limited sensor data and inferring between the lines... of what the velocity field should look like to be consistent with your measurement data and to be consistent with your governing equations"

- PINNs may struggle with certain systems (discontinuities, chaotic flows)
  - > "this is going to have a harder time on systems that are discontinuous with things like shock waves or chaotic convecting flows... it's not going to probably be amazing at those kinds of flows"

- Training PINNs can be challenging and often requires careful tuning
  - > "These don't always train that well sometimes they're stiff sometimes they overfit there's issues"

## Technical terms
- **[[Physics-Informed Neural Networks (PINNs)]]**: Neural networks that incorporate physical laws as constraints in the loss function
  - > "it's based on a kind of neural network idea that also incorporates known physics in the form of a partial differential equation in the loss function"

- **[[Automatic differentiation]]**: The computational technique used to calculate derivatives in neural networks
  - > "you can compute these terms essentially using the same kind of auto differentiation automatic differentiation and back propagation that would you would normally use to train a neural network"

- **[[Fractional PINNs]]**: An extension of PINNs that can handle fractional derivatives and integral terms
  - > "one of the ones I think is really cool are fractional PINNs for partial differential equations that have things like fractional derivatives... often times you have these nasty integral differential equations with fractional derivatives in your physics"

- **[[Delta PINNs]]**: A PINN variant that incorporates knowledge about the geometry of the problem domain
  - > "if you have a geometry... where my PDE is living you can actually bake into the PINNs... the eigenfunctions of the Laplace-Beltrami operator on that geometry give you a very good natural coordinate system in which to represent solution functions of that PDE"

## Predictions

- Traditional simulation methods will increasingly be supplemented by PINNs for certain applications
  - > "in the past we used to use a method called 4D VAR... it was this horrific physics-based constrained optimization... now in PINNs is kind of abstracted from the human, it uses modern machine learning architectures and optimizations and it typically does as well in a lot of cases"

## Surprises

- PINNs can effectively reconstruct fields from visual data not just numerical measurements
  - > "you can take very limited information and sometimes information that is not even like a velocity field like a smoke visualization or some kind of movie and you can infer what the actual velocity field that is kind of closest to satisfying the Navier-Stokes equations"

- The physics loss in PINNs can actually make optimization more difficult in some cases
  - > "they found that the loss function by increasing or decreasing that PDE-based soft constraint actually makes it more complex to optimize and harder to find a good solution in some cases"

## Conclusion

Physics-Informed Neural Networks represent an elegant approach at the intersection of machine learning and physics, allowing physics knowledge to guide neural network training through differential constraints. They excel at reconstructing fields from sparse data but require careful implementation. The method has seen widespread adoption and numerous extensions due to its intuitive nature and the ability to work with limited training data, though training challenges remain an active research area.

# AI/ML+Physics Part 1: Choosing what to model [Physics Informed Machine Learning]

![Thumbnail](https://img.youtube.com/vi/ARMk955pGbg/maxresdefault.jpg)

👤 [Steve Brunton](https://www.youtube.com/channel/UCm5mt-A4w61lknZ9lCsZtBw)  🔗 [Watch video](https://www.youtube.com/watch?v=ARMk955pGbg&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa&index=2&t=1s&pp=iAQB)
## Summary

This lecture focuses on the critical first stage of physics-informed machine learning: deciding what problem to actually model. The speaker emphasizes that this is perhaps the most foundational and important step, requiring the same rigor as traditional scientific and engineering design processes. The lecture covers various applications where machine learning can benefit physics problems, from discovering new physics to accelerating expensive simulations, while cautioning against using machine learning as a default solution when simpler methods would suffice.

> "computers are useless they can only give you answers and it's kind of the exact same story with machine learning okay these datadriven models are useless they can only you know model your data you have to be the one to decide what is the right model what is the right Fidelity what would be a useful model and how are you going to use it Downstream that's a human endeavor"

## Key Points

- **Problem definition requires iterative refinement across all ML stages**
  > "sometimes I don't know exactly what the problem is I have a vague notion like I Want to Build a Better race car or I want to build a better you know a better wing and that's not really a specific enough problem... very often we find oursel going down uh this path even all the way down youve trained your model and you realize it doesn't do exactly what you wanted it to"

- **Four main reasons to use ML in physics applications**
  > "learning new physics there are many many systems where we have a pretty good understanding of how the system works... but there are tons of systems where we don't know the physics so if I think about um Neuroscience we don't actually have governing equations for the brain or an epidemiological system or even the climate system"

- **Automatic differentiability provides unique advantages for engineering design**
  > "machine learning models are often not always but often kind of inherently automatically differentiable... we might be able to use this automatic differentiation that we're already using to train things like neural networks we might be able to use that to do better faster cheaper uh design optimization"

- **Chaotic systems require different modeling approaches**
  > "because the system is chaotic if I have a small uncertainty in my initial Condition it's going to massively amplify uh in the future so getting a deterministic prediction of the behavior of the system forward in time might be too much to ask for maybe what I want is a probability of being at some location in a future time"

## Technical Terms

- **[[Reynolds Average Navier Stokes (RANS)]]**: A turbulence modeling approach where Reynolds stresses require closure models
  > "there's a turbulence model called Reynolds average nav Stokes... these Reynold dresses in yellow we don't have really really good models of those terms we have to approximate them that's the closure problem"

- **[[Digital Twin]]**: A digital representation of a physical asset comprising multiple model fidelities
  > "the digital twin uh comprises a hierarchy of models at different fidelities some will be you know really crude simulations... some will be uh High Fidelity models that are really really expensive and we might have these machine learning models kind of in the middle uh stitching together these different fidelities"

- **[[Super Resolution]]**: Technique for enhancing low-resolution data to high-resolution using statistical patterns
  > "super resolution is an idea from image Sciences where you have a low reses image uh and with kind of the statistical information you've collected over lots and lots of data you might be able to learn how to fill in that low reses image into a highres image"

## Predictions

- **Benchmark systems will become critical for physics-ML**
  > "we need to have uh Benchmark systems kind of like image net for you know image classification we need similar benchmark works for dynamical systems for engineering systems for Control Systems"

- **Multi-fidelity surrogate models will enable better design optimization**
  > "you would hope that your surrogate machine learning model can take all of that data and build a model that's a little bit of The Best of Both Worlds you know low cost lower cost and lower error"

## Surprises

- **Problem definition may be the most important stage despite seeming simple**
  > "at first I actually thought this was going to be the easiest video to make um because you know how complicated could it be... but it turns out I've actually been spending the last couple of hours thinking about this and every time I think I have uh figured it out there's something new I want to tell you because this is actually maybe the most important out of all of these stages"

- **Turbulence closure modeling hit a wall that ML could overcome**
  > "turbulence closure modeling Reynolds average Navy or Stokes modeling is something where you know we had a huge amount of progress up in until you know the 1970s and 80s... at some point researchers hit a wall because these are really nasty you know functions to approximate... that's the perfect example of what machine learning is good at"

- **The choice between learning differential equations vs. discrete time steppers is fundamental**
  > "even that is a is a choice you have to make and there's a huge uh difference between these so the entire resnet architecture lives over here and the neural OD architecture lives over here they're solving related but different problems"

## Conclusion

The lecture establishes that problem formulation in physics-informed machine learning requires the same rigor as traditional scientific inquiry. The speaker emphasizes that machine learning should be viewed as a tool to augment physics understanding rather than replace it, with specific applications in discovering new physics, accelerating expensive simulations, and enabling better design optimization through automatic differentiability. The key insight is that proper problem definition often determines success more than the choice of architecture or optimization method.

# AI/ML+Physics Part 2: Curating Training Data [Physics Informed Machine Learning]

![Thumbnail](https://img.youtube.com/vi/g-S0m2zcKUg/maxresdefault.jpg)

👤 [Steve Brunton](https://www.youtube.com/channel/UCm5mt-A4w61lknZ9lCsZtBw)  🔗 [Watch video](https://www.youtube.com/watch?v=g-S0m2zcKUg&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa&index=3&pp=iAQB)
## Summary

This video discusses the second stage of physics-informed machine learning: data curation. The speaker covers various aspects of incorporating physics knowledge into the data collection and preparation process, emphasizing that this approach can significantly reduce data requirements compared to traditional machine learning methods.

The speaker argues that while companies like Google and Microsoft can rely on massive datasets to implicitly learn physics, most engineering applications need explicit physics integration to work with limited, expensive data:

> "if we embed physics into our learning process we can often get away with a lot less data... if I can incorporate physics to constrain that neural network to reduce the search space in that neural network I can often get away with much less data"

## Key Points

- **Data augmentation through physics symmetries**
  - > "if I think that this fluid system that I'm trying to model is invariant to rotations like if I rotate things nothing should change then in principle I can augment my data to include rotated copies and that enriches my data"

- **Critical importance of coordinate systems**
  - > "the coordinate system in which you measure your data really really really really matters and that is an opportunity for your human expertise for your physical knowledge and intuition to play a huge role"
  - > "getting your measurements right and in the right coordinate system is a huge Head Start that you can give your machine learning model"

- **Simulation vs experimental data tradeoffs**
  - > "often times in simulations I have a lot more spatial information I have high resolution spatial fields that are very hard to get in experiments... whereas if I had a wind tunnel experiment I might not get the full fluid flow field but I can run this thing for hours and hours and hours"
  - > "experiments have the real nitty-gritty details that we don't understand... the gold standard is still building experiments because that's where you're going to see if your assumptions broke"

- **Design optimization requires extrapolation beyond training data**
  - > "for design optimization if I want to design something better than my training data then I need my models my machine learning models to generalize beyond my training data"
  - > "the way you get your machine learning model to generalize beyond the training data... is by baking in physics into those models"

- **Data bias and rare events challenges**
  - > "if I have training data of ocean waves you know 99.999% of them are not rogue waves... I can build an extremely accurate machine learning model just by focusing on this boring data here and just ignoring the existence of these rare events"

- **Hidden variables in physical systems**
  - > "most of the systems we interact with we don't have access to all of the variables that are important for that system and we have to build models on partial information"

## Technical Terms

- **[[Physics-informed machine learning]]**: Integration of physical knowledge and constraints into machine learning models to improve performance with limited data
  - > "this introductory series on physics informed machine learning where we're essentially looking at the different opportunities and subtleties of incorporating physics into the machine learning process"

- **[[Data augmentation]]**: Artificially expanding datasets by applying known physical transformations
  - > "I can augment my data with copies of the data that I pass through those different Transformations"

- **[[Digital twin]]**: Machine learning models of physical assets used for optimization and design
  - > "digital models of physical assets so we can do improved optimization and design and control"

- **[[Multi-fidelity data]]**: Combining data sources of different accuracy and computational cost
  - > "lots of kind of in between areas where you might have multi Fidelity data sources or data from simulations and experiments"

## Predictions

- **Future of complex system design**
  - > "this is increasingly going to be how we design really complex systems in the future... it's going to give us the best of both worlds lower cost and lower error"

- **Active learning integration with digital twins**
  - > "if it has a high uncertainty but the design is promising it can actually go back and collect more data and it can collect data at different fidelities based on its uncertainty"

## Surprises

- **Historical physics discoveries involved strategic data selection**
  - > "presumably Galileo picked two densities that were dense enough that this could be kind of neglected and that was a very strategic human guided decision not to drop a beach ball but to drop denser objects"
  - > "if you actually drop balls of different densities and you start to see these fluid effects you'll actually find that the more classic and incorrect model of Aristotle... actually fits the data better"

- **Small signals can contain fundamental physics**
  - > "Einstein's relativity is only required for capturing that tiny last little bit so if I'm training the machine learning model and it's just trying to minimize error or loss it's almost certainly going to pick Newton's second law"

- **Single measurements can reconstruct full system states**
  - > "from a single measurement a single scalar measurement of the x coordinate of this Lorent system we can start learning how to embed that through some neural network coordinate transformation in a way that we get a simple explanatory differential equation out"

## Conclusion

The speaker emphasizes that data curation in physics-informed ML is highly problem-dependent and requires careful consideration of coordinate systems, data sources, biases, and the ultimate application goals. The key insight is that incorporating physics knowledge at the data stage can dramatically reduce data requirements and improve model generalization, which is crucial for engineering applications where data is expensive and design objectives often lie outside the training distribution.

# AI/ML+Physics Part 3: Designing an Architecture [Physics Informed Machine Learning]

![Thumbnail](https://img.youtube.com/vi/fiX8c-4K0-Q/maxresdefault.jpg)

👤 [Steve Brunton](https://www.youtube.com/channel/UCm5mt-A4w61lknZ9lCsZtBw)  🔗 [Watch video](https://www.youtube.com/watch?v=fiX8c-4K0-Q&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa&index=4&pp=iAQB)
## Summary

This lecture introduces Stage 3 of physics-informed machine learning: designing architectures. The speaker defines physics in terms of interpretability, generalizability, parsimony/simplicity, and symmetries/invariances/conservation laws, then explores how neural network architectures can embed these physical principles. The discussion covers various architectures from ResNets to Physics-Informed Neural Networks (PINNs), emphasizing how architectural choices constrain the space of possible functions and can enforce physical properties like Galilean invariance or energy conservation.

> "architectures Define a space of functions we're searching over and we find the function we want by tuning these free parameters Theta"

## Key Points

- **Physics definition for ML context**: Physics is characterized by four key properties
  > "interpretable and generalizable interpretable in the sense that they're usually very very simple... and it's very generalizable because the same physics that describes this apple also describes you know the physics of launching a rocket uh from the Earth to the Moon"

- **Architecture as function space constraint**: ML architectures fundamentally constrain the space of possible functions
  > "what most machine learning architectures do is they constrain the space of possible functions that could describe this input output relationship through a choice of architecture"

- **Parsimony principle**: Simplicity as a core physics principle dating back millennia
  > "this principle of Simplicity or parsimony has been the gold standard in physics for 2,000 years from Aristotle to Einstein the models that are more beautiful more parsimonious as simple as possible and no simpler typically encapsulate the core bits of physics"

- **Architecture-loss function coupling**: Architectures and loss functions are inherently intertwined
  > "architectures usually have loss functions that are good uh to train those architectures... loss functions often rely on an architecture and architectures often have custom loss functions you use to train those models"

- **Galilean invariance through architecture**: Julia Ling's work shows how architectural choices can enforce physical symmetries
  > "through a choice of this architecture with this custom tenser layer enforces Galilean invariance by construction all of the models represented in this function space are Galilean invariant"

## Technical Terms

- **[[Galilean Invariance]]**: A fundamental physics principle where the laws of physics remain unchanged in any inertial reference frame
  > "the physics doesn't change in any inertial reference frame so if I have a box of turbulence here or I have a box of turbulence moving at a constant velocity the physics the the closure model terms the physics shouldn't change in any inertial frame"

- **[[Invariance vs Equivariance]]**: Two distinct mathematical concepts for handling symmetries in ML
  > "invariance means that we know that the notion of a dog shouldn't matter if that dog is translated in the image or rotated or scaled... equivariance means that if I take my data and I transform it through some symmetry like a rotation some symmetry operation and then I run both of those through my neural network then the output of my neural network is also run through uh that rotation or translation"

- **[[SINDY (Sparse Identification of Nonlinear Dynamics)]]**: A non-neural network architecture for discovering differential equations
  > "Cindy the sparse identification of nonlinear Dynamics importantly this is not a neural network this is a generalized linear regression to learn a differential equation from data"

- **[[Physics-Informed Neural Networks (PINNs)]]**: Networks that incorporate physical laws through automatic differentiation
  > "you can take a normal feed forward Network that you would use to kind of predict those quantities and then because of the automatic differentiability uh of these neural network uh environments like py torch and tensor flow and Jacks you can often compute these partial derivatives of these quantities without having to like hard code it"

## Predictions

- **Extensive lecture series ahead**: The speaker plans substantial content on each architecture type
  > "we're going to have hours and hours and hours of you know material and lectures on various architectures... I'm pretty sure I have like 5 hours of material on Cindy alone"

- **Neuroscience-ML convergence**: Increasing integration between neuroscience and machine learning architectures
  > "these two fields are definitely evolving and growing together uh both neuroscience and machine learning"

## Surprises

- **Non-neural architectures included**: SINDY is presented as an architecture despite being generalized linear regression
  > "Cindy the sparse identification of nonlinear Dynamics importantly this is not a neural network this is a generalized linear regression to learn a differential equation from data okay and this is an architecture"

- **Wikipedia physics definition inadequacy**: The speaker found the standard physics definition insufficient for ML contexts
  > "I was at um a NPS Workshop about a month ago and I was giving a talk about you know machine learning for scientific discovery and I decided I should probably you know Wikipedia what is the definition of physics before I say that we're doing physics inform formed machine learning... that's fine and good but I don't like that as a working definition"

- **Historical pattern of simplification**: Scientific progress consistently leads to simpler, not more complex descriptions
  > "every time we've made this huge kind of Leap Forward in our understanding of physics things have actually gotten simpler the descriptions have gotten simpler and more Universal"

## Conclusion

The lecture establishes architectures as a critical stage in physics-informed machine learning, where the choice of network structure inherently constrains the space of learnable functions. By carefully designing architectures that embed physical principles like symmetries, conservation laws, and parsimony, researchers can create models that require less training data and generalize better. The upcoming series promises deep dives into specific architectures, with particular emphasis on how symmetries and invariances can be built into machine learning models by construction rather than learned from data augmentation.

# AI/ML+Physics Part 4: Crafting a Loss Function [Physics Informed Machine Learning]

![Thumbnail](https://img.youtube.com/vi/3SNkQ8jhKXc/maxresdefault.jpg)

👤 [Steve Brunton](https://www.youtube.com/channel/UCm5mt-A4w61lknZ9lCsZtBw)  🔗 [Watch video](https://www.youtube.com/watch?v=3SNkQ8jhKXc&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa&index=5&pp=iAQB)
## Summary

This video covers physics-informed machine learning specifically focusing on how to craft loss functions that embed physical knowledge into machine learning models. The presenter explains this as the fourth stage in a five-step physics-informed ML pipeline, emphasizing that custom loss functions are one of the most accessible ways to make ML models more physical, improving their generalization, learning efficiency, and sample efficiency.

> "this is one of the areas that's kind of most commonly used and it's one of the easiest ways of baking physics into a machine learning model which should improve its generalization capabilities uh its learning rates and efficiency sample efficiency lots of benefits and it's a relatively simple straightforward thing you can do to your machine learning model to make it more physical"

## Key Points

- **Physics-Informed Neural Networks (PINNs) add physics constraints through dual loss functions**
  - > "what pins physics and form neural networks does that's very very clever is they add a second loss function so because of the automatic differentiability uh of these modern machine learning uh environments P torch Jacks uh tensorflow so on you can take these quantities UV WP and you can compute their partial derivatives with respect to space and time"

- **PINNs balance data fitting with physics equation satisfaction**
  - > "pins adds a loss function it adds an extra loss function here which essentially says how much is the governing physical equation the partial differential equation that governs the physics how accurate is it how much is it violated"

- **Architecture and loss function choices are deeply interconnected**
  - > "this is a point I'm going to make over and over again architectures and loss functions really go hand in hand so this is an architecture you you are learning a lran you know with these inputs that's an architecture but the thing that allows you to train this architecture and have it make sense and be lran is that you have this loss function"

- **L1 and L2 norms promote different types of physical behavior**
  - > "we can promote models that are more low dimensional with the two Norm uh and we'll show you how this goes into the loss function in a minute and we can promote models that are simpler have less terms describing them or are more sparse using the one Norm"

- **Physical models should follow the principle of parsimony**
  - > "your machine learning model should be made as simple as possible to describe the data and no simpler this has been the gold standard of what is physics uh for 2,000 years from Aristotle to Einstein"

## Technical Terms

- **[[Physics-Informed Neural Networks (PINNs)]]**: Neural networks that incorporate physical equations as additional loss terms
  - > "physics informed neural network or a pin PN uh developed by uh ryy paricus and carneia dois"

- **[[SINDy (Sparse Identification of Nonlinear Dynamics)]]**: Method for discovering governing equations from data using sparse regression
  - > "I might also want to learn the Dynamics in that Laten space I want to might learn how to those coordinates evolve in time in which case I might use something like the sparse identification of nonlinear Dynamics to find the fewest terms in a differential equation"

- **[[Lagrangian Neural Networks]]**: Networks that learn Lagrangian functions and enforce Euler-Lagrange equations
  - > "if you know that your system conserves energy like it's a mechanical system like your double pendulum um we know that it does that by either having some lran structure you know it satisfies the oiler lrange equations"

- **[[L1 and L2 Norms]]**: Mathematical measures used to promote sparsity (L1) and smoothness/low-dimensionality (L2)
  - > "the L2 Norm measures distance in a typical ukian way it's just like the distance between two points as the crow flies the L1 Norm is a little bit of a different uh measure of distance sometimes we call it the the Manhattan Norm or the taxi cab Norm"

## Predictions

- **PINNs will remain popular due to their simplicity**
  - > "it's probably one of the most popular uh physics and machine learning algorithms out there cuz all you have to do is add this loss function uh and your models become more physical"

## Surprises

- **PINNs never exactly satisfy physics constraints**
  - > "the downside is that by adding this physics as a term in the loss function is that you're never really going to exactly satisfy that this loss is zero so an actual physical system an actual fluid flow this purple loss should be exactly zero"

- **Complex loss functions require extensive trial and error**
  - > "this was not an easy loss function to to come up with this took like months of trial and error figuring out how to actually craft the loss function to quantify what we meant by physical in this in this context"

- **Standard least squares regression produces overly complex physics models**
  - > "I'm going to get a physics model x dot that has 81 terms in the differential equation Y Dot's going to have 81 terms Z dots going to have 81 terms you never open up a physics book and see you know a model that has 81 terms in the differential equation"

## Conclusion

The video demonstrates that crafting appropriate loss functions is crucial for embedding physical knowledge into machine learning models. While methods like PINNs offer accessible entry points, creating truly effective physics-informed loss functions often requires sophisticated understanding of both the underlying physics and optimization principles. The presenter emphasizes that the next stage - optimization - will provide even more rigorous ways of enforcing physical constraints, moving from promoting physics to actually enforcing it.

# AI/ML+Physics Part 5: Employing an Optimization Algorithm [Physics Informed Machine Learning]

![Thumbnail](https://img.youtube.com/vi/T4iJ10TAIMg/maxresdefault.jpg)

👤 [Steve Brunton](https://www.youtube.com/channel/UCm5mt-A4w61lknZ9lCsZtBw)  🔗 [Watch video](https://www.youtube.com/watch?v=T4iJ10TAIMg&list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBa&index=6&pp=iAQB)
## Summary


> https://www.youtube.com/playlist?list=PLMrJAkhIeNNQ0BaKuBKY43k4xMo6NSbBaThis lecture covers the fifth stage of physics-informed machine learning: optimization algorithms for training models. The speaker emphasizes that physics can be embedded directly into optimization procedures through constrained optimization, which enforces physics exactly rather than just promoting it through loss functions.

> "if we only added this as a term in the loss function these two terms would be battling each other and you might get a model that's not exactly energy conserving and and uh doesn't have great model error because they're going to be kind of fighting each other but when you do this constrained optimization procedure when you build it directly into the optimization procedure you are minimizing the error and exactly satisfying those constraints"

## Key Points

- **Constrained vs. Penalized Optimization**: Direct constraints in optimization are superior to adding physics terms to loss functions
  > "with a loss function you're not exactly satisfying your constraints with constrained optimization you are exactly satisfying your constraints"

- **Energy Conservation in Fluid Dynamics**: JC Loiseau showed that incompressible fluid flows have 10 constraint equations that must be satisfied for energy conservation
  > "because of the incompressibility of this fluid flow from first principles we can derive a set of constraint equations that have to be true for any model that we get of this in compressible fluid flow these 10 constraint equations should be satisfied for energy conservation to hold"

- **Physics-Informed DMD**: Peter Baddoo developed methods to constrain optimization to search over matrices with specific symmetries (Hermitian, self-adjoint, etc.)
  > "what Peter Badu's physics informed DMD does is it essentially changes the space of matrices that our algorithm is searching over and so this is a constrained optimization where we are essentially constraining our search to a particular manifold of matrices"

- **Sparsity and Low-Dimensionality**: Physics models should be "as simple as possible to describe the data and no simpler," requiring custom optimization algorithms
  > "physics models models that capture the essential physic of physics of a system tend to be as simple as possible at to describe the data and no simpler"

## Technical Terms

- **[[SINDy]]**: Sparse Identification of Nonlinear Dynamics - a procedure for discovering differential equations from data
  > "this Cindy sparse model identification procedure where we try to find the fewest Columns of theta that add up to equal x dot y Dot and Z dot"

- **[[KKT Constrained Least Squares]]**: Karush-Kuhn-Tucker optimization method for exactly satisfying equality constraints
  > "through this KK ke kkt constrainedly squares he's able to exactly satisfy the constraints and minimize this model error"

- **[[Procrustes Problem]]**: Mathematical optimization for restricting solutions to matrix manifolds
  > "the mathematical optimization problem that allows us to to restrict to these Matrix manifolds is called a procris problem"

- **[[SR3 Algorithm]]**: A new sparse optimization algorithm that finds solutions more efficiently than traditional methods
  > "using this new kind of sr3 algorithm you almost immediately go to the optimal solution and it's better conditioned"

## Predictions

- Custom optimization algorithms will become increasingly important for physics-informed ML
  > "sometimes you'll actually need custom algorithms to promote um... sometimes you're going to need custom algorithms to promote a certain type of constraint or to work with a certain type of loss function"

## Surprises

- Many constrained optimization problems have exact closed-form solutions, not just approximate ones
  > "often times submanifold constraints often also have Exact Solutions so it's not out of the question that you could solve both of these you know uh by designing an optimization algorithm to keep you on this submanifold or Subspace"

- Symbolic regression uses fundamentally different optimization (evolutionary algorithms) compared to typical ML
  > "this is a whole different set of optimization algorithms that is you know designed to guide this architecture to find the right model"

## Conclusion

The speaker argues that embedding physics directly into optimization algorithms through constrained optimization is the "gold standard" for physics-informed machine learning, though it requires more human effort than simply adding physics terms to loss functions. This approach guarantees exact satisfaction of physical constraints rather than just promoting them.