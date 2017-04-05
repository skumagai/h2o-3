package hex.schemas;

import hex.klime.KLime;
import hex.klime.KLimeModel;
import water.api.API;
import water.api.schemas3.ModelParametersSchemaV3;

public class KLimeV3 extends ModelBuilderSchema<KLime, KLimeV3, KLimeV3.KLimeParametersV3> {
  public static final class KLimeParametersV3 extends ModelParametersSchemaV3<KLimeModel.KLimeParameters, KLimeParametersV3> {
    public static String[] fields = new String[] {
            "model_id",
            "training_frame",
            "k",
            "seed",
    };

    @API(help = "FIXME", direction = API.Direction.INOUT, gridable = true)
    public int k;

    @API(help = "Seed for pseudo random number generator (if applicable)", gridable = true)
    public long seed;

  }
}
