#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* define convenience macros */
#define streq(s1,s2)    (!strcmp ((s1), (s2)))

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

/* todo: eventually remove */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_CHANNELS                    STBI_grey


typedef enum {
    TF_ALLOCATE_TENSOR              = 10, // Error allocating tensors.
    TF_RESIZE_TENSOR                = 11, // Error resizing tensor.
    TF_ALLOCATE_TENSOR_AFTER_RESIZE = 12, // Error allocating tensors after resize.
    TF_COPY_BUFFER_TO_INPUT         = 13, // Error copying input from buffer.
    TF_INVOKE_INTERPRETER           = 14, // Error invoking interpreter.
    TF_COPY_OUTOUT_TO_BUFFER        = 15, // Error copying output to buffer.
} errorCode;


// Dispose of the model and interpreter objects.
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

int infer(uint8_t *img_buffer, char *model_filename, int input_xsize, int input_ysize)
{
    /* the image size */
    int image_size = input_xsize * input_ysize * IMAGE_CHANNELS;

    /* return codes to track TFLite C API errors */
    TfLiteStatus tfl_status;

    /* load the model */
    TfLiteModel  *model = TfLiteModelCreateFromFile(model_filename);

    /* create the interpreter */
    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, NULL);

    /* Allocate tensors */
    tfl_status = TfLiteInterpreterAllocateTensors(interpreter);

    /* Log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
      disposeTfLiteObjects(model, interpreter);
      return TF_ALLOCATE_TENSOR;
    }

    int input_dims[4] = {1, input_ysize, input_xsize, IMAGE_CHANNELS};
    tfl_status = TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, 4);

    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
      disposeTfLiteObjects(model, interpreter);
      return TF_RESIZE_TENSOR;
    }

    tfl_status = TfLiteInterpreterAllocateTensors(interpreter);

    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
      disposeTfLiteObjects(model, interpreter);
      return TF_ALLOCATE_TENSOR_AFTER_RESIZE;
    }

    /* The input tensor */
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    /* Copy the JPG image data into into the input tensor */
    tfl_status = TfLiteTensorCopyFromBuffer(input_tensor, img_buffer, image_size * sizeof(uint8_t));
    
    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
      disposeTfLiteObjects(model, interpreter);
      return TF_COPY_BUFFER_TO_INPUT;
    }

    /* invoke interpreter */
    tfl_status = TfLiteInterpreterInvoke(interpreter);

    /* log and exit in case of error */
    if(tfl_status != kTfLiteOk)
    {
      disposeTfLiteObjects(model, interpreter);
      return TF_INVOKE_INTERPRETER;
    }

    /* extract the output tensor data */
    const TfLiteTensor* output_tensor_boxes = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    const TfLiteTensor* output_tensor_classes = TfLiteInterpreterGetOutputTensor(interpreter, 1);
    const TfLiteTensor* output_tensor_score = TfLiteInterpreterGetOutputTensor(interpreter, 2);
    const TfLiteTensor* output_tensor_count = TfLiteInterpreterGetOutputTensor(interpreter, 3);

    /* todo: do the doings */

    /* dispose of the TensorFlow objects */
    disposeTfLiteObjects(model, interpreter);
}


int main(int argc, char *argv [])
{
    /* return code to track general errors */
    int rc = 0;

    /* get provider host and port from command arguments */
    int argv_index_input = -1;
    int argv_index_model = -1;
    int argv_index_xsize = -1;
    int argv_index_ysize = -1;

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
    int img_xsize_training = atoi(argv[argv_index_xsize]);
    int img_ysize_training = atoi(argv[argv_index_ysize]);

    // get the model filename 

    /* decode the image file */
    uint8_t *img_buffer = (uint8_t*)stbi_load(argv[argv_index_input], &img_xsize, &img_ysize, &img_channels_infile, IMAGE_CHANNELS);

    /* initialize resized image buffer to the training input size */
    int img_buffer_training_input_size = img_xsize_training * img_ysize_training * IMAGE_CHANNELS;
    uint8_t img_buffer_resized[img_buffer_training_input_size];

    /* downsample the image i.e., resize the image to a smaller dimension */
    rc = stbir_resize_uint8(img_buffer, img_xsize, img_ysize, 0, img_buffer_resized, img_xsize_training, img_ysize_training, 0, IMAGE_CHANNELS);

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
        rc = stbi_write_jpg("test/training_input.jpg", img_xsize_training, img_ysize_training, IMAGE_CHANNELS, (void *)img_buffer_resized, img_quality_desired);

        /* the stb_write functions returns 0 on failure and non-0 on success. */
        if(rc == 0)
        {
            /* print error message in case of error */
            printf("Error code %d when attempting to write the resized image\n", rc);
        }
        else
        {
            /* success message */
            printf("Resized: %d x %d --> %d x %d\n", img_xsize, img_ysize, img_xsize_training, img_ysize_training);
        } 
    }
#endif


    /* inference */
    infer(img_buffer, argv[argv_index_model], img_xsize_training, img_ysize_training);
    
    /* free the input image data buffer */
    stbi_image_free(img_buffer);

    /* end program */
    return 0;
}