#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* tensorflow header summary: */
/* https://github.com/tensorflow/tensorflow/blob/b5ee46383fe070c6c678a4fbacca63bfea202aa1/tensorflow/lite/c/README.md#header-summary= */
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"


/* define convenience macros */
#define streq(s1,s2)    (!strcmp ((s1), (s2)))

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_CHANNELS        STBI_rgb


typedef enum {
    TF_ALLOCATE_TENSOR              = 10, /* error allocating tensors */
    TF_RESIZE_TENSOR                = 11, /* error resizing tensor */
    TF_ALLOCATE_TENSOR_AFTER_RESIZE = 12, /* error allocating tensors after resize */
    TF_COPY_BUFFER_TO_INPUT         = 13, /* error copying input from buffer */
    TF_INVOKE_INTERPRETER           = 14, /* error invoking interpreter */
    TF_COPY_OUTOUT_TO_BUFFER        = 15, /* error copying output to buffer */
} errorCode;


/* dispose of the model and interpreter objects */
int disposeTfLiteObjects(TfLiteModel* pModel, TfLiteInterpreter* pInterpreter)
{
    if(pModel != NULL)
    {
        TfLiteModelDelete(pModel);
    }

    if(pInterpreter)
    {
        TfLiteInterpreterDelete(pInterpreter);
    }
}


/* inference/prediction function */
int infer(uint8_t *img_buffer, char *model_filename, int input_xsize, int input_ysize, int input_mean, int input_std)
{
    /* the image size */
    int image_size = input_xsize * input_ysize * IMAGE_CHANNELS;

    /* return codes to track TFLite C API errors */
    TfLiteStatus tfl_status;

    /* load the model */
    TfLiteModel  *model = TfLiteModelCreateFromFile(model_filename);

    /* create the interpreter */
    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, NULL);

    /* allocate tensors */
    tfl_status = TfLiteInterpreterAllocateTensors(interpreter);

    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
        disposeTfLiteObjects(model, interpreter);
        printf("ERROR: %d\n", TF_ALLOCATE_TENSOR);
        return TF_ALLOCATE_TENSOR;
    }

    int input_dims[4] = {1, input_ysize, input_xsize, IMAGE_CHANNELS};
    tfl_status = TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, 4);

    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
        disposeTfLiteObjects(model, interpreter);
        printf("ERROR: %d\n", TF_RESIZE_TENSOR);
        return TF_RESIZE_TENSOR;
    }

    tfl_status = TfLiteInterpreterAllocateTensors(interpreter);

    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
        disposeTfLiteObjects(model, interpreter);
        printf("ERROR: %d\n", TF_ALLOCATE_TENSOR_AFTER_RESIZE);
        return TF_ALLOCATE_TENSOR_AFTER_RESIZE;
    }

    /* the input tensor */
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    /* the array that will collect the JPEG's RGB values */
    /* the TfLiteTensorCopyFromBuffer function expect buffer of type float */
    float img_buffer_rescaled[image_size];

    /* todo: make rescaling work, causes TfLiteTensorCopyFromBuffer to return an error code */
    /* this is because the trained model expects uint values from 0 to 255 (and not floats) */
    /* the RGB range is 0-255. Rescale it based on given input mean and standard deviation */
    /* e.g. with input mean 0 and input std 255 the 0-255 RGB range is rescaled to 0-1 */
    /*
    for(int i = 0; i < image_size; i++)
    {
        img_buffer_rescaled[i] = ((float)img_buffer[i] - input_mean) / input_std;
    }*/

    /* copy the JPG image data into into the input tensor */
    tfl_status = TfLiteTensorCopyFromBuffer(input_tensor, img_buffer, image_size * sizeof(uint8_t));
    
    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
        disposeTfLiteObjects(model, interpreter);
        printf("ERROR: %d\n", TF_COPY_BUFFER_TO_INPUT);
        return TF_COPY_BUFFER_TO_INPUT;
    }

    /* invoke interpreter */
    tfl_status = TfLiteInterpreterInvoke(interpreter);

    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
        disposeTfLiteObjects(model, interpreter);
        printf("ERROR: %d\n", TF_INVOKE_INTERPRETER);
        return TF_INVOKE_INTERPRETER;
    }

    /* extract the output tensor data */

    /* total bounding box count */
    // const TfLiteTensor *output_tensor_count = TfLiteInterpreterGetOutputTensor(interpreter, 3);  
    // float count_buffer[1];
    // TfLiteStatus status = TfLiteTensorCopyToBuffer(output_tensor_count, count_buffer, sizeof(float));
    // int count = (int)count_buffer[0];
    // printf("count: %i\n", count);

    /* bounding boxes */
    const TfLiteTensor *output_tensor_boxes = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    float boxes[count][4];
    TfLiteTensorCopyToBuffer(output_tensor_boxes, boxes, count * 4 * sizeof(float));

    const TfLiteTensor *output_tensor_score = TfLiteInterpreterGetOutputTensor(interpreter, 2);
    float scores_buffer[count];
    TfLiteTensorCopyToBuffer(output_tensor_score, scores_buffer, count * sizeof(float));

    for (int i = 0; i < count; i++)
    {
        printf("%f,%f,%f,%f,%f\n", scores_buffer[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]);
    }

    /* labels (there is only 1 label...) */
    //const TfLiteTensor *output_tensor_classes = TfLiteInterpreterGetOutputTensor(interpreter, 1);

    /* dispose of the TensorFlow objects */
    disposeTfLiteObjects(model, interpreter);
}


/* main function */
int main(int argc, char *argv [])
{
    /* return code to track general errors */
    int rc = 0;

    /* get provider host and port from command arguments */
    int argv_index_input = -1;
    int argv_index_model = -1;
    int argv_index_xsize = -1;
    int argv_index_ysize = -1;
    int argv_index_mean  = -1;
    int argv_index_std   = -1;

    // --------------------------------------------------------------------------
    // parse the command arguments (all arguments are optional)

    int argn;
    for (argn = 1; argn < argc; argn++)
    {
        if (streq (argv [argn], "--help")
        ||  streq (argv [argn], "-?"))
        {
            printf("inference [options] ...");
            printf("\n  --input    / -i        input image filename");
            printf("\n  --model    / -m        tflite model filename");
            printf("\n  --xsize    / -x        training input width");
            printf("\n  --ysize    / -y        training input height");
            printf("\n  --mean     / -n        input mean (optional - not supported)");
            printf("\n  --std      / -s        input standard deviation (optional - not supported)");
            printf("\n  --help     / -?        this information\n\n");
            
            /* program exit code */
            return 1;
        }
        else
        if (streq (argv [argn], "--input")
        ||  streq (argv [argn], "-i"))
            argv_index_input = ++argn;
        else
        if (streq (argv [argn], "--model")
        ||  streq (argv [argn], "-m"))
            argv_index_model = ++argn;
        else
        if (streq (argv [argn], "--xsize")
        ||  streq (argv [argn], "-x"))
            argv_index_xsize = ++argn;
        else
        if (streq (argv [argn], "--ysize")
        ||  streq (argv [argn], "-y"))
            argv_index_ysize = ++argn;
        else
        if (streq (argv [argn], "--mean")
        ||  streq (argv [argn], "-n"))
            argv_index_mean = ++argn;
        else
        if (streq (argv [argn], "--std")
        ||  streq (argv [argn], "-s"))
            argv_index_std = ++argn;
        else
        {
            /* print error message */
            printf("Unknown option: %s\n\n", argv[argn]);

            /* program exit code */
            return 1;
        }
    }

    /* check if mandatory arguments were provided */
    if (argv_index_input < 0 || argv_index_model < 0 || argv_index_xsize < 0 || argv_index_ysize < 0)
    {
        printf("Missing options, try: ./inference --help\n");

        /* program exit code */
        return 1;
    }

    /* the values of these properties will be set from reading the input image when invoking stbi_load */
    int img_xsize;
    int img_ysize;
    int img_channels_infile;

    /* target properties for the resized image */
    int img_xsize_model_input = atoi(argv[argv_index_xsize]);
    int img_ysize_model_input = atoi(argv[argv_index_ysize]);

    /* mean and standard deviation, use default values if not specified */
    int input_mean = argv_index_mean < 0 ? 0 : atoi(argv[argv_index_mean]);
    int input_std = argv_index_std < 0 ? 1 : atoi(argv[argv_index_std]);

    /* get the model filename */ 

    /* decode the image file */
    uint8_t *img_buffer = (uint8_t*)stbi_load(argv[argv_index_input], &img_xsize, &img_ysize, &img_channels_infile, IMAGE_CHANNELS);

    /* initialize resized image buffer to the training input size */
    int img_buffer_training_input_size = img_xsize_model_input * img_ysize_model_input * IMAGE_CHANNELS;
    uint8_t img_buffer_resized[img_buffer_training_input_size];

    /* downsample the image i.e., resize the image to a smaller dimension */
    rc = stbir_resize_uint8(img_buffer, img_xsize, img_ysize, 0, img_buffer_resized, img_xsize_model_input, img_ysize_model_input, 0, IMAGE_CHANNELS);

    /* returned result is 1 for success or 0 in case of an error */
    if(rc == 0)
    {   
        /* print error message in case of error */
        printf("Error code %d when attempting to resize the image\n", rc);
    }

#if 0
    else
    {
        /* image successfully resized in memory */
        /* write image to file */
        /* todo: eventually remove */
        int img_quality_desired = 90;
        rc = stbi_write_jpg("test/training_input.jpg", img_xsize_model_input, img_ysize_model_input, IMAGE_CHANNELS, (void *)img_buffer_resized, img_quality_desired);

        /* the stb_write functions returns 0 on failure and non-0 on success. */
        if(rc == 0)
        {
            /* print error message in case of error */
            printf("Error code %d when attempting to write the resized image\n", rc);
        }
        else
        {
            /* success message */
            printf("Resized: %d x %d --> %d x %d\n", img_xsize, img_ysize, img_xsize_model_input, img_ysize_model_input);
        } 
    }
#endif


    /* inference */
    infer(img_buffer_resized, argv[argv_index_model], img_xsize_model_input, img_ysize_model_input, input_mean, input_std);
    
    /* free the input image data buffer */
    stbi_image_free(img_buffer);

    /* end program */
    return 0;
}