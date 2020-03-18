import sbt.Resolver

organization  := "ch.unibas.cs.gravis"

name := "HelloWorld"

version       := "0.1"

scalaVersion  := "2.12.8"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

resolvers ++= Seq(
  Resolver.bintrayRepo("unibas-gravis", "maven"),
  Resolver.bintrayRepo("cibotech", "public"),
  Opts.resolver.sonatypeSnapshots
)


libraryDependencies  ++= Seq(
            "ch.unibas.cs.gravis" % "scalismo-native-all" % "4.0.+",
            "ch.unibas.cs.gravis" %% "scalismo-ui" % "0.14-RC1",
	    "ch.unibas.cs.gravis" %% "scalismo-faces" % "0.10.1",
            "com.cibo" %% "evilplot" % "0.6.3"
)

assemblyJarName in assembly := "HelloWorld.jar"

mainClass in assembly := Some("example.ExampleApp")

assemblyMergeStrategy in assembly :=  {
    case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
    case PathList("META-INF", s) if s.endsWith(".SF") || s.endsWith(".DSA") || s.endsWith(".RSA") => MergeStrategy.discard
    case "reference.conf" => MergeStrategy.concat
    case _ => MergeStrategy.first
}
