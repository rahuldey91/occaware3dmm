import java.io.File

import scalismo.faces.color.{RGB, RGBA}
import scalismo.faces.deluminate.SphericalHarmonicsOptimizer
import scalismo.faces.image.{AccessMode, LabeledPixelImage, PixelImage, PixelImageOperations}
import scalismo.faces.io.{PixelImageIO, TLMSLandmarksIO}
import scalismo.faces.parameters.{ParametricRenderer, RenderParameter, SphericalHarmonicsLight}
import scalismo.faces.sampling.face.evaluators.PixelEvaluators._
import scalismo.faces.sampling.face.evaluators.PointEvaluators.IsotropicGaussianPointEvaluator
import scalismo.faces.sampling.face.evaluators.PriorEvaluators.{GaussianShapePrior, GaussianTexturePrior}
import scalismo.faces.sampling.face.evaluators._
import scalismo.faces.sampling.face.proposals.ImageCenteredProposal.implicits._
import scalismo.faces.sampling.face.proposals.ParameterProposals.implicits._
import scalismo.faces.sampling.face.proposals.SphericalHarmonicsLightProposals._
import scalismo.faces.sampling.face.proposals._
import scalismo.faces.sampling.face.{MoMoRenderer, ParametricLandmarksRenderer, ParametricModel}
import scalismo.geometry.{Vector, Vector3D, _1D, _2D}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.proposals.MixtureProposal.implicits._
import scalismo.sampling.proposals.{MetropolisFilterProposal, MixtureProposal}
import scalismo.sampling._
import scalismo.utils.Random
import scalismo.faces.sampling.face.proposals.SphericalHarmonicsLightProposals.{RobustSHLightSolverProposal, RobustSHLightSolverProposalWithLabel}
import scalismo.faces.segmentation.LoopyBPSegmentation
import scalismo.faces.segmentation.LoopyBPSegmentation.{BinaryLabelDistribution, Label, LabelDistribution}
import scalismo.faces.gui._
import scalismo.faces.gui.GUIBlock._
import scalismo.faces.io.MoMoIO
import scalismo.faces.sampling.face.ParametricImageRenderer
import scalismo.faces.sampling.face.loggers.PrintLogger
import faces.sampling.face.proposals.SegmentationMasterProposal

object ExampleApp {

    def main(args: Array[String]) {

    	scalismo.initialize()
	val seed = 1986L
	implicit val rnd = Random(seed)
  }
}
