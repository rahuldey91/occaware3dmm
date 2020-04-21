
package faces.apps

import java.io.File

import scalismo.color.{RGB, RGBA}
import scalismo.faces.deluminate.SphericalHarmonicsOptimizer
import scalismo.faces.image.{AccessMode,  PixelImage}
import scalismo.faces.io.{PixelImageIO, TLMSLandmarksIO}
import scalismo.faces.parameters.{RenderParameter}
import scalismo.faces.sampling.face.evaluators.PixelEvaluators._
import scalismo.faces.sampling.face.evaluators.PointEvaluators.IsotropicGaussianPointEvaluator
import scalismo.faces.sampling.face.evaluators.PriorEvaluators.{GaussianShapePrior, GaussianTexturePrior}
import scalismo.faces.sampling.face.evaluators._
import scalismo.faces.sampling.face.proposals.ImageCenteredProposal.implicits._
import scalismo.faces.sampling.face.proposals.ParameterProposals.implicits._
import scalismo.faces.sampling.face.proposals.SphericalHarmonicsLightProposals._
import scalismo.faces.sampling.face.proposals._
import scalismo.faces.sampling.face.{MoMoRenderer, ParametricLandmarksRenderer, ParametricModel}
import scalismo.geometry.{EuclideanVector, EuclideanVector3D, _1D, _2D}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.proposals.MixtureProposal.implicits._
import scalismo.sampling.proposals.{MetropolisFilterProposal, MixtureProposal}
import scalismo.sampling._
import scalismo.utils.Random
import scalismo.faces.sampling.face.proposals.SphericalHarmonicsLightProposals.{RobustSHLightSolverProposalWithLabel}
import scalismo.faces.segmentation.LoopyBPSegmentation
import scalismo.faces.segmentation.LoopyBPSegmentation.{BinaryLabelDistribution, Label, LabelDistribution}
import scalismo.faces.gui._
import scalismo.faces.gui.GUIBlock._
import scalismo.faces.io.MoMoIO
import scalismo.faces.sampling.face.ParametricImageRenderer
import scalismo.faces.sampling.face.loggers.PrintLogger
import scalismo.faces.sampling.face.proposals.SegmentationMasterProposal



/* This Fitscript with its evaluators and the proposal distribution follows closely the proposed setting of:

Occlusion-aware 3D Morphable Models and an Illumination Prior for Face Image Analysis
Bernhard Egger, Sandro Schönborn, Andreas Schneider, Adam Kortylewski, Andreas Morel-Forster, Clemens Blumer and Thomas Vetter
International Journal of Computer Vision (IJCV), 2018
DOI: https://doi.org/10.1007/s11263-018-1064-8
To understand the concepts behind the fitscript and the underlying methods there is a tutorial on:
http://gravis.dmi.unibas.ch/pmm/
 */

object OcclusionFitScript{
  def defaultPoseProposal(lmRenderer: ParametricLandmarksRenderer)(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    import MixtureProposal.implicits._

    val yawProposalC = GaussianRotationProposal(EuclideanVector3D.unitY, 0.75f)
    val yawProposalI = GaussianRotationProposal(EuclideanVector3D.unitY, 0.10f)
    val yawProposalF = GaussianRotationProposal(EuclideanVector3D.unitY, 0.01f)
    val rotationYaw = MixtureProposal(0.1 *: yawProposalC + 0.4 *: yawProposalI + 0.5 *: yawProposalF)

    val pitchProposalC = GaussianRotationProposal(EuclideanVector3D.unitX, 0.75f)
    val pitchProposalI = GaussianRotationProposal(EuclideanVector3D.unitX, 0.10f)
    val pitchProposalF = GaussianRotationProposal(EuclideanVector3D.unitX, 0.01f)
    val rotationPitch = MixtureProposal(0.1 *: pitchProposalC + 0.4 *: pitchProposalI + 0.5 *: pitchProposalF)

    val rollProposalC = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.75f)
    val rollProposalI = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.10f)
    val rollProposalF = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.01f)
    val rotationRoll = MixtureProposal(0.1 *: rollProposalC + 0.4 *: rollProposalI + 0.5 *: rollProposalF)

    val rotationProposal = MixtureProposal(0.5 *: rotationYaw + 0.3 *: rotationPitch + 0.2 *: rotationRoll).toParameterProposal

    val translationC = GaussianTranslationProposal(EuclideanVector(300f, 300f)).toParameterProposal
    val translationF = GaussianTranslationProposal(EuclideanVector(50f, 50f)).toParameterProposal
    val translationHF = GaussianTranslationProposal(EuclideanVector(10f, 10f)).toParameterProposal
    val translationProposal = MixtureProposal(0.2 *: translationC + 0.2 *: translationF + 0.6 *: translationHF)

    val distanceProposalC = GaussianDistanceProposal(500f, compensateScaling = true).toParameterProposal
    val distanceProposalF = GaussianDistanceProposal(50f, compensateScaling = true).toParameterProposal
    val distanceProposalHF = GaussianDistanceProposal(5f, compensateScaling = true).toParameterProposal
    val distanceProposal = MixtureProposal(0.2 *: distanceProposalC + 0.6 *: distanceProposalF + 0.2 *: distanceProposalHF)

    val scalingProposalC = GaussianScalingProposal(0.15f).toParameterProposal
    val scalingProposalF = GaussianScalingProposal(0.05f).toParameterProposal
    val scalingProposalHF = GaussianScalingProposal(0.01f).toParameterProposal
    val scalingProposal = MixtureProposal(0.2 *: scalingProposalC + 0.6 *: scalingProposalF + 0.2 *: scalingProposalHF)

    val poseMovingNoTransProposal = MixtureProposal(rotationProposal + distanceProposal + scalingProposal)
    val centerREyeProposal = poseMovingNoTransProposal.centeredAt("right.eye.corner_outer", lmRenderer).get
    val centerLEyeProposal = poseMovingNoTransProposal.centeredAt("left.eye.corner_outer", lmRenderer).get
    val centerRLipsProposal = poseMovingNoTransProposal.centeredAt("right.lips.corner", lmRenderer).get
    val centerLLipsProposal = poseMovingNoTransProposal.centeredAt("left.lips.corner", lmRenderer).get

    MixtureProposal(centerREyeProposal + centerLEyeProposal + centerRLipsProposal + centerLLipsProposal + 0.2 *: translationProposal)
  }


  /* Collection of all statistical model (shape, texture) related proposals */
  def neutralMorphableModelProposal(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {

    val shapeC = GaussianMoMoShapeProposal(0.2f)
    val shapeF = GaussianMoMoShapeProposal(0.1f)
    val shapeHF = GaussianMoMoShapeProposal(0.025f)
    val shapeScaleProposal = GaussianMoMoShapeCaricatureProposal(0.2f)
    val shapeProposal = MixtureProposal(0.1f *: shapeC + 0.5f *: shapeF + 0.2f *: shapeHF + 0.2f *: shapeScaleProposal).toParameterProposal

    val textureC = GaussianMoMoColorProposal(0.2f)
    val textureF = GaussianMoMoColorProposal(0.1f)
    val textureHF = GaussianMoMoColorProposal(0.025f)
    val textureScale = GaussianMoMoColorCaricatureProposal(0.2f)
    val textureProposal = MixtureProposal(0.1f *: textureC + 0.5f *: textureF + 0.2 *: textureHF + 0.2f *: textureScale).toParameterProposal

    MixtureProposal(shapeProposal + textureProposal )
  }

  /* Collection of all statistical model (shape, texture, expression) related proposals */
  def defaultMorphableModelProposal(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {


    val expressionC = GaussianMoMoExpressionProposal(0.2f)
    val expressionF = GaussianMoMoExpressionProposal(0.1f)
    val expressionHF = GaussianMoMoExpressionProposal(0.025f)
    val expressionScaleProposal = GaussianMoMoExpressionCaricatureProposal(0.2f)
    val expressionProposal = MixtureProposal(0.1f *: expressionC + 0.5f *: expressionF + 0.2f *: expressionHF + 0.2f *: expressionScaleProposal).toParameterProposal


    MixtureProposal(neutralMorphableModelProposal + expressionProposal)
  }

  /* Collection of all color transform proposals */
  def defaultColorProposal(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    val colorC = GaussianColorProposal(RGB(0.01f, 0.01f, 0.01f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
    val colorF = GaussianColorProposal(RGB(0.001f, 0.001f, 0.001f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
    val colorHF = GaussianColorProposal(RGB(0.0005f, 0.0005f, 0.0005f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))

    MixtureProposal(0.2f *: colorC + 0.6f *: colorF + 0.2f *: colorHF).toParameterProposal
  }

  /* Collection of all illumination related proposals */
  def illuminationProposal(modelRenderer: ParametricImageRenderer[RGBA] with ParametricModel, target: PixelImage[RGBA])(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {

    val lightSHPert = SHLightPerturbationProposal(0.001f, fixIntensity = true)
    val lightSHIntensity = SHLightIntensityProposal(0.1f)

    val lightSHBandMixter = SHLightBandEnergyMixer(0.1f)
    val lightSHSpatial = SHLightSpatialPerturbation(0.05f)
    val lightSHColor = SHLightColorProposal(0.01f)

    MixtureProposal(lightSHSpatial + lightSHBandMixter + lightSHIntensity + lightSHPert + lightSHColor).toParameterProposal
  }

  def segmentLBP(target: PixelImage[RGBA],
                 current: (RenderParameter, PixelImage[Int]),
                 renderer: MoMoRenderer)(implicit rnd: Random): (PixelImage[LabelDistribution])= {


    val curSample: PixelImage[RGBA] = renderer.renderImage(current._1)

    val nonfaceHist = HistogramRGB.fromImageRGBA(target, 25, 0)
    val nonfaceProb: PixelImage[Double] = target.map(p => nonfaceHist.logValue(p.toRGB))

    val maskedTarget = PixelImage(curSample.domain, (x, y) => RGBA(target(x, y).toRGB, curSample(x, y).a * current._2(x, y).toFloat))
    val fgHist = HistogramRGB.fromImageRGBA(maskedTarget, 25, 0) // replace by hist defined on foreground

    val sdev = 0.043f
    val pixEvalHSV: IsotropicGaussianPixelEvaluatorHSV = IsotropicGaussianPixelEvaluatorHSV(sdev)
    val neighboorhood = 4
    var x: Int = 0
    val fgProbBuffer = PixelImage(nonfaceProb.domain, (x, y) => 0.0).toBuffer

    val curSampleR = curSample.withAccessMode(AccessMode.Repeat())
    while (x < target.width) {
      var y: Int = 0
      while (y < target.height) {
        if (curSample(x, y).a > 0) {
          var maxNeigboorhood = Double.NegativeInfinity
          var q: Int = -neighboorhood
          while (q <= neighboorhood) {
            // Pixels in Face Region F
            var p: Int = -neighboorhood
            while (p <= neighboorhood) {
              val t1 = pixEvalHSV.logValue(target(x, y).toRGB, curSampleR(x + p, y + q).toRGB)
              maxNeigboorhood = Math.max(t1, maxNeigboorhood)
              p += 1
            }
            q += 1
          }
          fgProbBuffer(x, y) = maxNeigboorhood
        }
        else {
          // Pixels in Nonface Region B
          fgProbBuffer(x, y) = fgHist.logValue(target(x, y).toRGB)
        }
        y += 1
      }
      x += 1
    }
    val fgProb: PixelImage[Double] = fgProbBuffer.toImage

    val imageGivenFace = fgProb.map(p => math.exp(p))
    val imageGivenNonFace = nonfaceProb.map(p => math.exp(p))

    val numLabels = 2
    def binDist(pEqual: Double, numLabels: Int, width: Int, height: Int): BinaryLabelDistribution = {
      val pElse = (1 - pEqual) / (numLabels - 1)
      def binDist(k: Label, l: Label) = if (k == l) pEqual else pElse
      PixelImage.view(width, height, (x, y) => binDist)
    }
    val smoothnessDistribution = binDist(0.9, numLabels, target.width, target.height)

    val init = PixelImage(target.width, target.height, (x, y) => Label(0))

    LoopyBPSegmentation.segmentImageFromProb(target.map(_.toRGB), init, imageGivenNonFace, imageGivenFace, smoothnessDistribution, numLabels, 5, false)
  }

  def binDist(pEqual: Double, numLabels: Int, width: Int, height: Int): BinaryLabelDistribution = {
    val pElse = (1 - pEqual) / (numLabels - 1)
    def binDist(k: Label, l: Label) = if (k == l) pEqual else pElse
    PixelImage.view(width, height, (x, y) => binDist)
  }

  def fit(targetFn : String, lmFn: String, outputDir: String, modelRenderer: MoMoRenderer, expression: Boolean = true)(implicit rnd: Random): (RenderParameter, PixelImage[Int]) = {
    val target = PixelImageIO.read[RGBA](new File(targetFn)).get
    val targetLM = TLMSLandmarksIO.read2D(new File(lmFn)).get.filter(lm => lm.visible)
//    ImagePanel(target).displayIn("Target Image")

    // Building the proposal model
    /* Collection of all pose related proposals */

    // pose proposal
    val totalPose = defaultPoseProposal(modelRenderer)

    //light proposals
    val lightProposal = illuminationProposal(modelRenderer, target)

    //color proposals
    val colorProposal = defaultColorProposal

    //Morphable Model  proposals
    val expression = true
    val momoProposal = if(expression) defaultMorphableModelProposal else neutralMorphableModelProposal

    // Landmarks Evaluator
    val pointEval = IsotropicGaussianPointEvaluator[_2D](4.0) //lm click uncertainty in pixel! -> should be related to image/face size
    val landmarksEval = LandmarkPointEvaluator(targetLM, pointEval, modelRenderer)

    // Prior Evaluator
    val priorEval = ProductEvaluator(GaussianShapePrior(0, 1), GaussianTexturePrior(0, 1))


    // full proposal filtered by the landmark and prior Evaluator
    val proposal = MetropolisFilterProposal(MetropolisFilterProposal(MixtureProposal(totalPose + colorProposal + 3f*:momoProposal+ 2f *: lightProposal), landmarksEval), priorEval)


    // Evaluator
    val sdev = 0.043f
    val faceEval = IsotropicGaussianPixelEvaluator(sdev)
    val nonFaceEval = HistogramRGB.fromImageRGBA(target, 25)
    val imgEval = LabeledIndependentPixelEvaluator(target, faceEval, nonFaceEval)
    val labeledModelEval = LabeledImageRendererEvaluator(modelRenderer, imgEval)

    // a dummy segmentation proposal
    class SegmentationProposal(implicit rnd: Random) extends ProposalGenerator[(RenderParameter, PixelImage[Int])]  with SymmetricTransitionRatio[(RenderParameter, PixelImage[Int])] {
      override def propose(current: (RenderParameter, PixelImage[Int])): (RenderParameter, PixelImage[Int]) = {current}
    }


    // a joint proposal for $\theta$ and $z$ (in this implementation the segmentation proposal is never chosen)
    val masterProposal = SegmentationMasterProposal(proposal, new SegmentationProposal, 1)
    val printLogger = PrintLogger[RenderParameter](Console.out, "").verbose
    val imageFitter = MetropolisHastings(masterProposal, labeledModelEval)

    val poseFitter = MetropolisHastings(totalPose, landmarksEval)

    //landmark chain for initialisation
    val initDefault: RenderParameter = RenderParameter.defaultSquare.fitToImageSize(target.width, target.height)
    val init50 = initDefault.withMoMo(initDefault.momo.withNumberOfCoefficients(50, 50, 5))
    val initLMSamples: IndexedSeq[RenderParameter] = poseFitter.iterator(init50).take(5000).toIndexedSeq

    val lmScores = initLMSamples.map(rps => (landmarksEval.logValue(rps), rps))

    val bestLM = lmScores.maxBy(_._1)._2
    val imgLM = modelRenderer.renderImage(bestLM)
//    ImagePanel(imgLM).displayIn("Pose Initialization")

    val shOpt = SphericalHarmonicsOptimizer(modelRenderer, target)
    val robustShOptimizerProposal = RobustSHLightSolverProposalWithLabel(modelRenderer, shOpt, target, iterations = 100)
    val dummyImg = target.map(_ => 0)
    val robust = robustShOptimizerProposal.propose(bestLM, dummyImg)

    val robustImg = modelRenderer.renderImage(robust._1)
    val consensusSet = robust._2.map(RGB(_))

//    shelf(
//      ImagePanel(robustImg),
//      ImagePanel(consensusSet)
//    ).displayIn("Robust Illumination Estimation")

    val labeledPrintLogger = PrintLogger[(RenderParameter, PixelImage[Int])](Console.out, "")//.verbose
    var first1000 = imageFitter.iterator(robust, labeledPrintLogger).take(1000).toIndexedSeq.last
    val first1000Img = modelRenderer.renderImage(first1000._1)
//    shelf(
//      ImagePanel(first1000Img)
//    ).displayIn("After first 1000 samples")

    val nonfaceHist = HistogramRGB.fromImageRGBA(target, 25, 0)
    val nonfaceProb: PixelImage[Double] = target.map(p => nonfaceHist.logValue(p.toRGB))

    val maskedTarget = PixelImage(first1000Img.domain, (x, y) => RGBA(target(x, y).toRGB, first1000Img(x, y).a * first1000._2(x, y).toFloat))
    val fgHist = HistogramRGB.fromImageRGBA(maskedTarget, 25, 0) // replace by hist defined on foreground



    val pixEvalHSV: IsotropicGaussianPixelEvaluatorHSV = IsotropicGaussianPixelEvaluatorHSV(sdev)
    val neighboorhood = 4
    var x: Int = 0
    val fgProbBuffer = PixelImage(nonfaceProb.domain, (x, y) => 0.0).toBuffer
    val first1000ImgR = first1000Img.withAccessMode(AccessMode.Repeat())
    while (x < target.width) {
      var y: Int = 0
      while (y < target.height) {
        if (first1000Img(x, y).a > 0) {
          var maxNeigboorhood = Double.NegativeInfinity
          var q: Int = -neighboorhood
          while (q <= neighboorhood) {
            // Pixels in Face Region F
            var p: Int = -neighboorhood
            while (p <= neighboorhood) {
              val t1 = pixEvalHSV.logValue(target(x, y).toRGB, first1000ImgR(x + p, y + q).toRGB)
              maxNeigboorhood = Math.max(t1, maxNeigboorhood)
              p += 1
            }
            q += 1
          }
          fgProbBuffer(x, y) = maxNeigboorhood
        }
        else {
          // Pixels in Nonface Region B
          fgProbBuffer(x, y) = fgHist.logValue(target(x, y).toRGB)
        }
        y += 1
      }
      x += 1
    }
    val fgProb: PixelImage[Double] = fgProbBuffer.toImage

    // converting to log likelihoods
    val imageGivenFace = fgProb.map(p => math.exp(p))
    val imageGivenNonFace = nonfaceProb.map(p => math.exp(p))

    val numLabels = 2

    val smoothnessDistribution = binDist(0.9, numLabels, target.width, target.height)

    val init = PixelImage(target.width, target.height, (x, y) => Label(0))

    LoopyBPSegmentation.segmentImageFromProb(target.map(_.toRGB), init, imageGivenNonFace, imageGivenFace, smoothnessDistribution, numLabels, 5, true)

    var current = first1000
    var i = 0

    while (i < 5) {

      // segment
      val zLabel = segmentLBP(target, current, modelRenderer)

      // update state
      current = (current._1, zLabel.map(_.maxLabel))

      // fit and update state
      current = imageFitter.iterator(current, labeledPrintLogger).take(1000).toIndexedSeq.last

      // visualize fit and label
      val zLabelImg = zLabel.map (l => RGB(l(Label(1))))
      val thetaImg = modelRenderer.renderImage(current._1)
//      shelf(
//        ImagePanel(zLabelImg),
//        ImagePanel(thetaImg)
//      ).displayIn("After Iteration: " + i)
      i += 1
    }
    current

  }
}



object Occlusion3DMM extends App {
  scalismo.initialize()
  val seed = 1986L
  implicit val rnd = Random(seed)
  println("start")

  // Load face model and target image
  val modelface12 = MoMoIO.read(new File("data/model2017-1_face12_nomouth.h5")).get
  val rendererFace12 =  MoMoRenderer(modelface12, RGBA.BlackTransparent).cached(5)

  val targetFn = "data/fit.png"
  val lmFn = "data/fit.tlms"
  val outDir = "data/out/"

  val fit = OcclusionFitScript.fit(targetFn, lmFn, outDir, rendererFace12,true)
  PixelImageIO.write(fit._2.map(p => if(1 == p) RGB.White else RGB.Black), new File(outDir + "finalSegmentation.png"))
  PixelImageIO.write(rendererFace12.renderImage(fit._1), new File(outDir + "finalFit.png"))

  println("stop")
}