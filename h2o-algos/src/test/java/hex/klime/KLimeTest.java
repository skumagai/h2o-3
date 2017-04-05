package hex.klime;

import org.junit.BeforeClass;
import org.junit.Test;
import water.DKV;
import water.Key;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class KLimeTest extends TestUtil {

  @BeforeClass()
  public static void setup() { stall_till_cloudsize(1); }

  @Test
  public void testTitanic() throws Exception {
    Scope.enter();
    try {
      Frame fr = loadTitanicData();
      Frame expected = Scope.track(parse_test_file("smalldata/klime_test/titanic_surr_preds.csv"));

      KLimeModel.KLimeParameters p = new KLimeModel.KLimeParameters();
      p._seed = 12345;
      p._k = 12;
      p._ignored_columns = new String[]{"PassengerId", "Survived", "predict", "p0"};
      p._train = fr._key;
      p._response_column = "p1";

      KLimeModel klm = (KLimeModel) Scope.track_generic(new KLime(p).trainModel().get());

      Frame scored = Scope.track(klm.score(fr));

      assertEquals(1, scored.numCols());
      assertVecEquals(expected.vec(0), scored.vec(0), 0.0001);

      // 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
      Map<Integer, List<KLimeModel.ReasonCode>> codes1 = klm.explain(new double[]{2, 1, 22, 1, 0});
      assertEquals(1, codes1.size());

      // 2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
      Map<Integer, List<KLimeModel.ReasonCode>> codes2 = klm.explain(new double[]{0, 0, 38, 1, 0});
      assertEquals(1, codes2.size());
    } finally {
      Scope.exit();
    }
  }

  private static Frame loadTitanicData() {
    Key<Frame> titanic = Key.<Frame>make("titanic");
    Frame fr = Scope.track(parse_test_file(titanic, "smalldata/klime_test/titanic_preds.csv"));
    fr.replace(0, fr.vec(0).toCategoricalVec());
    DKV.put(fr);
    return fr;
  }

}