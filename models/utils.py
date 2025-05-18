import models.model_1
import models.model_2
import models.model_3
import models.model_4
import models.model_5
import models.model_6


def get_model_class(model_name):
    model_classes_dict = {'model_1': models.model_1.Model,
                          'model_2': models.model_2.Model,
                          'model_3': models.model_3.Model,
                          'model_4': models.model_4.Model,
                          'model_5': models.model_5.Model,
                          'model_6': models.model_6.Model}

    return model_classes_dict[model_name]
