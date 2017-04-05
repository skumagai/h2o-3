package hex.klime;

import hex.*;

import hex.glm.GLMModel;
import hex.klime.KLimeModel.*;
import water.*;
import water.fvec.Chunk;
import water.fvec.Frame;

import java.util.*;

import static hex.kmeans.KMeansModel.KMeansParameters;

public class KLimeModel extends Model<KLimeModel, KLimeParameters, KLimeOutput> {

  public KLimeModel(Key<KLimeModel> selfKey, KLimeParameters params, KLimeOutput output) {
    super(selfKey, params, output);
    assert(Arrays.equals(_key._kb, selfKey._kb));
  }

  @Override
  public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    return new ModelMetricsRegression.MetricBuilderRegression();
  }

  public Map<Integer, List<ReasonCode>> explain(double[] data) {
    // FIXME
    double[] data_backup = data.clone();
    int cluster = (int) _output._clustering.score1(data, new double[1])[0];
    data = data_backup;

    return Collections.singletonMap(cluster, explain(cluster, data));
  }

  public List<ReasonCode> explain(int cluster, double[] data) {
    GLMModel model = (GLMModel) _output._regressionModels[cluster];
    DataInfo dinfo = model.dinfo();

    final double[] b = model.beta();
    final String[] coeffNames = model._output.coefficientNames();

    List<ReasonCode> codes = new LinkedList<>();
    for (int i = 0; i < dinfo._cats; i++) {
      int l = dinfo.getCategoricalId(i, data[i]);
      if (l >= 0) codes.add(new ReasonCode(coeffNames[l], b[l]));
    }
    int numStart = dinfo.numStart();
    int ncats = dinfo._cats;
    for (int i = 0; i < dinfo._nums; i++) {
      double d = data[ncats + i];
      if (! dinfo._skipMissing && Double.isNaN(d))
        d = dinfo._numMeans[i];
      codes.add(new ReasonCode(coeffNames[numStart + i], b[numStart + i] * d));
    }
    Collections.sort(codes, Collections.<ReasonCode>reverseOrder());

    return codes;
  }

  @Override
  public double[] score0(Chunk[] chks, double weight, double offset, int row_in_chunk, double[] tmp, double[] preds) {
    double[] ps = _output._clustering.score0(chks, weight, offset, row_in_chunk, tmp, preds);
    int cluster = (int) ps[0];
    if ((cluster < 0) || (cluster >= _output._regressionModels.length)) {
      throw new IllegalStateException("Unknown cluster, cluster id = " + cluster);
    }
    return _output._regressionModels[cluster].score0(chks, weight, offset, row_in_chunk, tmp, preds);
  }

  @Override
  protected double[] score0(double[] data, double[] preds) {
    throw H2O.unimpl("Intentionally not implemented, should never be called!");
  }

  @Override
  public double deviance(double w, double y, double f) {
    return (y - f) * (y - f);
  }

  public static class KLimeParameters extends Model.Parameters {
    public String algoName() { return "KLime"; }
    public String fullName() { return "k-LIME"; }
    public String javaName() { return KLimeModel.class.getName(); }

    public int _k;

    @Override public long progressUnits() { return fillClusteringParms(new KMeansParameters(), null).progressUnits() + _k; }

    KMeansParameters fillClusteringParms(KMeansParameters p, Key<Frame> clusteringFrameKey) {
      p._estimate_k = false;
      p._k = _k;
      p._train = clusteringFrameKey;
      p._auto_rebalance = false;
      p._seed = _seed;
      return p;
    }

    GLMModel.GLMParameters fillRegressionParms(GLMModel.GLMParameters p, Key<Frame> frameKey) {
      p._family = GLMModel.GLMParameters.Family.gaussian;
      p._alpha = new double[] {0.5};
      p._lambda_search = true;
      p._intercept = true;
      p._train = frameKey;
      p._response_column = _response_column;
      p._weights_column = "__cluster_weights";
      p._auto_rebalance = false;
      p._seed = _seed;
      return p;
    }
  }

  public static class KLimeOutput extends Model.Output {
    public KLimeOutput(KLime b) { super(b); }

    public Model _clustering;
    public Model[] _regressionModels;

    @Override public ModelCategory getModelCategory() { return ModelCategory.Regression; }
  }

  @Override
  protected Futures remove_impl(Futures fs) {
    if (_output._clustering != null)
      _output._clustering.remove(fs);
    if (_output._regressionModels != null)
      for (Model m : _output._regressionModels)
        m.remove(fs);
    return super.remove_impl(fs);
  }

  public static final class ReasonCode implements Comparable<ReasonCode> {
    public final String _code;
    public final double _coef;
    public ReasonCode(String _code, double _coef) {
      this._code = _code;
      this._coef = _coef;
    }
    @Override
    public int compareTo(ReasonCode o) { return Double.compare(_coef, o._coef); }
  }

}
