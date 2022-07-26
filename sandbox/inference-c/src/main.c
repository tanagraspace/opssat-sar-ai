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


int main(int argc, char *argv [])
{
    /* function return code to track errors */
    int rc = 0;

    /* get provider host and port from command arguments */
    int argv_index_input = -1;
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
    if (argv_index_input < 0 || argv_index_xsize < 0 || argv_index_ysize < 0)
    {
        printf("Missing options, try: ./inference --help\n");

        /* program exit code */
        return 1;
    }

    /* the values of these properties will be some from reading the input image when invoking stbi_load */
    int img_xsize;
    int img_ysize;
    int img_channels_infile;

    /* target properties for the resized image */
    int img_xsize_training = atoi(argv[argv_index_xsize]);
    int img_ysize_training = atoi(argv[argv_index_ysize]);

    /* decode the image file */
    uint8_t *img_buffer = (uint8_t*)stbi_load(argv[argv_index_input], &img_xsize, &img_ysize, &img_channels_infile, STBI_grey);

    /* initialize resized image buffer to the training input size */
    int img_buffer_training_input_size = img_xsize_training * img_ysize_training * STBI_grey;
    uint8_t img_buffer_resized[img_buffer_training_input_size];

    /* downsample the image i.e., resize the image to a smaller dimension */
    rc = stbir_resize_uint8(img_buffer, img_xsize, img_ysize, 0, img_buffer_resized, img_xsize_training, img_ysize_training, 0, STBI_grey);

    /* returned result is 1 for success or 0 in case of an error */
    if(rc == 0)
    {   
        /* print error message in case of error */
        printf("Error code %d when attempting to resize the image\n", rc);
    }
    else
    {
        /* image successfully resized in memory */
        /* write image to file */
        /* todo: eventually remove */
        int img_quality_desired = 90;
        rc = stbi_write_jpg(argv[argv_index_output], img_xsize_training, img_ysize_training, STBI_grey, (void *)img_buffer_resized, img_quality_desired);

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
    
    /* free the input image data buffer */
    stbi_image_free(img_buffer);

    /* end program */
    return 0;
}