package hex.klime;

import hex.Model;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.ModelMetrics;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.kmeans.KMeans;
import hex.kmeans.KMeansModel;
import water.*;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.FrameUtils;

import java.util.HashSet;
import java.util.Set;

import static hex.kmeans.KMeansModel.KMeansParameters;

public class KLime extends ModelBuilder<KLimeModel, KLimeModel.KLimeParameters, KLimeModel.KLimeOutput> {

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  public KLime(boolean startup_once) { super(new KLimeModel.KLimeParameters(), startup_once); }

  public KLime(KLimeModel.KLimeParameters parms) {
    super(parms);
    init(false);
  }

  @Override
  protected Driver trainModelImpl() {
    return new KLimeDriver();
  }

  private class KLimeDriver extends Driver {
    @Override
    public void computeImpl() {
      KLimeModel model = null;
      Set<Key<Frame>> frameKeys = new HashSet<>(_parms._k);
      try {
        init(true);

        // The model to be built
        model = new KLimeModel(dest(), _parms, new KLimeModel.KLimeOutput(KLime.this));
        model.delete_and_lock(_job);

        Key<Frame> clusteringTrainKey = Key.<Frame>make("klime_clustering_" + _parms._train);
        frameKeys.add(clusteringTrainKey);
        Frame clusteringTrain = new Frame(clusteringTrainKey);
        clusteringTrain.add(train());
        clusteringTrain.remove(_parms._response_column);
        DKV.put(clusteringTrain);

        KMeansParameters kmeansParms = _parms.fillClusteringParms(new KMeansParameters(), clusteringTrain._key);
        KMeans clustering = new KMeans(kmeansParms, _job);
        KMeansModel clusteringModel = clustering.trainModelNested(null);

        Frame clusterLabels = Scope.track(clusteringModel.score(clusteringTrain));

        final int K = clusteringModel._output._k[clusteringModel._output._iterations - 1];
        String[] clusterNames = new String[K];
        for (int i = 0; i < K; i++)
          clusterNames[i] = "cluster" + i;

        clusterLabels.vec(0).setDomain(clusterNames);
        DKV.put(clusterLabels);

        Frame clusterWeights = Scope.track(FrameUtils.categoricalEncoder(clusterLabels, new String[0],
                Model.Parameters.CategoricalEncodingScheme.OneHotExplicit, null));

        ModelBuilder[] glmBuilders = new ModelBuilder[K];
        for (int i = 0; i < K; i++) {
          Key<Frame> key = Key.<Frame>make("klime_train_cluster_" + i + _parms._train);
          frameKeys.add(key);
          Frame frame = new Frame(key);
          Vec weightVec = clusterWeights.vec(clusterWeights.find("predict." + clusterNames[i]));
          frame.add("__cluster_weights", weightVec);
          frame.add(train());
          DKV.put(frame);

          Key<Model> glmKey = Key.<Model>make("klime_glm_cluster_" + i + model._key);
          Job glmJob = new Job<>(glmKey, ModelBuilder.javaName("glm"), "k-LIME Regression (GLM, cluster = " + i + ")");
          DKV.put(glmJob);
          Scope.track_generic(glmJob);

          GLM glmBuilder = ModelBuilder.make("GLM", glmJob, glmKey);
          _parms.fillRegressionParms(glmBuilder._parms, key);

          glmBuilders[i] = glmBuilder;
        }

        bulkBuildModels(_job, glmBuilders, 1);

        Model[] regressionModels = new Model[K];
        for (int i = 0; i < glmBuilders.length; i++)
          regressionModels[i] = (GLMModel) DKV.getGet(glmBuilders[i]._job._result);

        model._output._clustering = clusteringModel;
        model._output._regressionModels = regressionModels;

        model.score(_parms.train()).delete(); // This scores on the training data and appends a ModelMetrics
        model._output._training_metrics = ModelMetrics.getFromDKV(model, _parms.train());

        model.update(_job);
      } finally {
        if (model != null) model.unlock(_job);
        Futures fs = new Futures();
        for (Key<Frame> k : frameKeys)
          DKV.remove(k, fs);
        fs.blockForPending();
      }
    }
  }

}