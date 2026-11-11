## Master's Thesis in Remote Sensing
This repository is an archive for Matteo Luelli's Master's Thesis with the title: Emulation of High‑Spectral‑Resolution Atmospheric Radiative Transfer for Imaging Spectroscopy Earth Observation Satellite Missions. The main objective was to design AI-based emulators to obtain better performance than Look-Up Tables (LUTs) interpolation of values physically simulated with models such as MODTRAN6, while keeping fast computational speeds. This would allow such emulators to be a valid alternative for hyperspectral missions such as ESA's FLEX. This task has to leverage synthetic high-resolution spectral datasets of atmospheric radiative transfer functions, generated starting from atmospheric and geometric parameters.

### Preliminary Dataset
The dataset employed for the thesis was originally generated and used in AMLEC (Atmospheric Machine Learning Emulation Challenge).

The data can be found at this link: https://zenodo.org/records/17670939

### Hardware Specifications
The following are the hardware specifications of the main computer used to run the code, obtain results and measure runtime:
<ul>
  <li><b>GPU</b>: NVIDIA GeForce RTX 4060 Ti (8 GB of VRAM)</li>
  <li><b>CPU</b>: AMD RYZEN 9 7900 (3.70 GHz, 12 cores)</li>
  <li><b>RAM</b>: 32 GB DDR5 with 4800 MT/s</li>
  <li><b>Storage</b>: SSD Crucial P310 1TB PCIe M.2</li>
</ul>

### Bibliography
[1] Alexander Berk, Lawrence S Bernstein, and David C Robertson. Modtran: A moderate resolution model for lowtran 7. Technical Report GL-TR-89-0122, Geophysics Laboratory, Air Force Systems Command, Hanscom AFB, MA, 1989.

[2] Alexander Berk, Patrick Conforti, Rosemary Kennett, Timothy Perkins, Fred Hawes, and Jeannette Van Den Bosch. Modtran® 6: A major upgrade of the modtran® radiative transfer code. In 2014 6th workshop on hyperspectral image and signal processing: Evolution in remote sensing (WHISPERS), pages 1–4. IEEE, 2014.

[3] Philip G Brodrick, David R Thompson, Jay E Fahlen, Michael L Eastwood, Charles M Sarture, Sarah R Lundeen, Winston Olson-Duvall, Nimrod Carmon, and Robert O Green. Generalized radiative transfer emulation for imaging spectroscopy reflectance retrievals. Remote Sensing of Environment, 261:112476, 2021.

[4] Brian D Bue, David R Thompson, Shubhankar Deshpande, Michael Eastwood, Robert O Green, Vijay Natraj, Terry Mullen, and Mario Parente. Neural network radiative transfer for imaging spectroscopy. Atmospheric Measurement Techniques, 12(4):2567–2578, 2019.

[5] Girish Chandrashekar and Ferat Sahin. A survey on feature selection methods. Computers & electrical engineering, 40(1):16–28, 2014.

[6] George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303–314, 1989.

[7] Matthias Drusch, Jose Moreno, Umberto Del Bello, Raffaella Franco, Yves Goulas, Andreas Huth, Stefan Kraft, Elizabeth M Middleton, Franco Miglietta, Gina Mohammed, et al. The fluorescence explorer mission concept-esa’s earth explorer 8. IEEE Transactions on Geoscience and Remote Sensing, 55(3):1273–1284, 2016.

[8] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7), 2011.

[9] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks, 107:3–11, 2018.

[10] Milton Friedman. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal of the american statistical association, 32(200):675–701, 1937.

[11] José Luis García-Soria, Miguel Morata, Katja Berger, Ana Belén Pascual-Venteo, Juan Pablo Rivera-Caicedo, and Jochem Verrelst. Evaluating epistemic uncertainty estimation strategies in vegetation trait retrieval using hybrid models and imaging spectroscopy data. Remote Sensing of Environment, 310:114228, 2024.

[12] Iouli E Gordon, Laurence S Rothman, Robert J Hargreaves, R Hashemi, Ekaterina Vladimirovna Karlovets, FM Skinner, Eamon K Conway, Christian Hill, Roman V Kochanov, Y Tan, et al. The hitran2020 molecular spectroscopic database. Journal of quantitative spectroscopy and radiative transfer, 277:107949, 2022.

[13] Iouli E Gordon, LS Rothman, Robert J Hargreaves, Frances M Gomez, Thibault Bertin, Christian Hill, Roman V Kochanov, Y Tan, P Wcisło, Vladimir Yu Makhnev, et al. The hitran2024 molecular spectroscopic database. Journal of Quantitative Spectroscopy and Radiative Transfer, 353:109807, 2026.

[14] Luis Guanter, Hermann Kaufmann, Karl Segl, Saskia Foerster, Christian Rogass, Sabine Chabrillat, Theres Kuester, André Hollstein, Godela Rossner, Christian Chlebek, et al. The enmap spaceborne imaging spectroscopy mission for earth observation. Remote Sensing, 7(7):8830–8857, 2015.

[15] Luis Guanter, Rudolf Richter, and Hermann Kaufmann. On the application of the modtran4 atmospheric radiative transfer code to optical remote sensing. International Journal of Remote Sensing, 30(6):1407–1424, 2009.

[16] Isabelle Guyon and André Elisseeff. An introduction to variable and feature selection. Journal of machine learning research, 3(Mar):1157–1182, 2003.

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[18] Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian journal of statistics, pages 65–70, 1979.

[19] Kurt Hornik. Approximation capabilities of multilayer feedforward networks. Neural networks, 4(2):251–257, 1991.

[20] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359–366, 1989.

[21] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7132–7141, 2018.

[22] Ian T Jolliffe and Jorge Cadima. Principal component analysis: a review and recent developments. Philosophical transactions. Series A, Mathematical, physical, and engineering sciences, 374(2065):20150202, 2016.

[23] George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6):422–440, 2021.

[24] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017.

[25] Serkan Kiranyaz, Onur Avci, Osama Abdeljaber, Turker Ince, Moncef Gabbouj, and Daniel J Inman. 1d convolutional neural networks and applications: A survey. Mechanical systems and signal processing, 151:107398, 2021.

[26] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

[27] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

[28] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–444, 2015.

[29] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1(4):541–551, 1989.

[30] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[31] Shunlin Liang, Xiaowen Li, and JindiWang. Chapter 5 - atmospheric correction of optical imagery. In Advanced Remote Sensing, pages 111–126. Academic Press, Boston, 2012.

[32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[33] Luca Martino, Roberto San Millán-Castillo, and Eduardo Morgado. Spectral information criterion for automatic elbow detection. Expert Systems with Applications, 231:120705, 2023.

[34] Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.

[35] Chuizheng Meng, Sam Griesemer, Defu Cao, Sungyong Seo, and Yan Liu. When physics meets machine learning: A survey of physics-informed machine learning. Machine Learning for Computational Science and Engineering, 1(1):20, 2025.

[36] Radford M Neal. Bayesian learning for neural networks. Springer Science & Business Media, 2012.

[37] Karl Pearson. Liii. on lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin philosophical magazine and journal of science, 2(11):559–572, 1901.

[38] Stefano Pignatti, Angelo Palombo, Simone Pascucci, Filomena Romano, Federico Santini, Tiziana Simoniello, Amato Umberto, Cuomo Vincenzo, Nicola Acito, Marco Diani, et al. The prisma hyperspectral mission: Science activities and opportunities for agriculture and land monitoring. In 2013 IEEE international geoscience and remote sensing symposium-IGARSS, pages 4558–4561. IEEE, 2013.

[39] EO Postma, HJ van den Herik, and LJ van der Maaten. Dimensionality reduction: a comparative review. Journal of Machine Learning Research, 10(1-41):66–71, 2009.

[40] Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, Cambridge, MA, 2006.

[41] Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain. Psychological review, 65(6):386, 1958.

[42] Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747, 2016.

[43] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533–536, 1986.

[44] Bernhard Schölkopf, Alexander Smola, and Klaus-Robert Müller. Nonlinear component analysis as a kernel eigenvalue problem. Neural computation, 10(5):1299–1319, 1998.

[45] Jorge Vicent Servera, Alexander A Kokhanovsky, Gherardo Varando, Luca Martino, and Gustau Camps-Valls. Symbolic regression for physics-aware emulation. IEEE Transactions on Geoscience and Remote Sensing, 64:1–13, 2026.

[46] Jorge Vicent Servera, Luca Martino, Jochem Verrelst, and Gustau Camps-Valls. Multifidelity gaussian process emulation for atmospheric radiative transfer models. IEEE Transactions on Geoscience and Remote Sensing, 61:1–10, 2023.

[47] Jorge Vicent Servera, Luca Martino, Jochem Verrelst, Juan Pablo Rivera-Caicedo, and Gustau Camps-Valls. Multioutput feature selection for emulation and sensitivity analysis. IEEE Transactions on Geoscience and Remote Sensing, 62:1–11, 2024.

[48] Jorge Vicent Servera, Juan Pablo Rivera-Caicedo, Jochem Verrelst, Jordi Muñoz-Marí, Neus Sabater, Béatrice Berthelot, Gustau Camps-Valls, and José Moreno. Systematic assessment of modtran emulators for atmospheric correction. IEEE Transactions on Geoscience and Remote Sensing, 60:1–17, 2021.

[49] Jorge Vicent Servera, Jasdeep Singh, Axel Rochel, Julio Cesar Contreras Huerta, Romain Poirier, Panagiotis Liatsis, Hasan Al Marzouqi, and Gustau Camps-Valls. Evaluating machine learning emulators for atmospheric radiative transfer: The atmospheric machine learning emulation challenge. IEEE Geoscience and Remote Sensing Magazine, 2026.

[50] Tijmen Tieleman. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2):26, 2012.

[51] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.

[52] Eric F Vermote, Didier Tanré, Jean L Deuze, Maurice Herman, and J-J Morcette. Second simulation of the satellite signal in the solar spectrum, 6s: An overview. IEEE transactions on geoscience and remote sensing, 35(3):675–686, 1997.

[53] Eric F. Vermote, Didier Tanré, Jean-Luc Deuzé, Michael Herman, Jean-Jacques Morcette, and Svetlana Y. Kotchenova. Second simulation of a satellite signal in the solar spectrum–vector (6sv): 6s user guide version 3. Technical report, Department of Geography, University of Maryland, 2006.

[54] Jorge Vicent, Jochem Verrelst, Juan Pablo Rivera-Caicedo, Neus Sabater, Jordi Munoz-Mari, Gustau Camps-Valls, and José Moreno. Emulation as an accurate alternative to interpolation in sampling radiative transfer codes. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 11(12):4918–4931, 2018.

[55] Jorge Vicent, Jochem Verrelst, Neus Sabater, Luis Alonso, Juan Pablo Rivera-Caicedo, Luca Martino, Jordi Muñoz-Marí, and José Moreno. Comparative analysis of atmospheric radiative transfer models using the atmospheric lookup table generator (alg) toolbox (version 2.0). Geoscientific model development, 13(4):1945–1957, 2020.

[56] Frank Wilcoxon. Individual comparisons by ranking methods. Biometrics bulletin, 1(6):80–83, 1945.