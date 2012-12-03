#include "cudamat.cuh"
#include "cudaconv2/conv_util.cuh"
#include "cudaconv2/cudaconv2.cuh"
#include "nvmatrix/nvmatrix.cuh"


/*
 * images:      (numImgColors, imgPixels, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)             if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModules, numImages)
 */
extern "C" void convUp(
		       cudamat* images,
		       cudamat* filters,
		       cudamat* targets,
		       int numModulesX,
		       int paddingStart,
		       int moduleStride,
		       int numImgColors, 
		       int numGroups
		       //float scaleTargets,
		       //float scaleOutput
		       ){

  // next, call it.
  _filterActsCu(images,
  		 filters,
  		 targets,
  		 numModulesX,
  		 paddingStart,
  		 moduleStride,
  		 numImgColors,
  		 numGroups, 0, 1, true) ;
    //scaleTargets,
    //scaleOutput);

}



extern "C" void convDown(
		       cudamat* images,
		       cudamat* filters,
		       cudamat* targets,

		       int imgSize,
		       int paddingStart,
		       int moduleStride,
		       int numImgColors,
		       int numGroups){

  _imgActsCu(images,
	      filters,
	      targets,

	      imgSize,
	      paddingStart,
	      moduleStride,
	      numImgColors,
	      numGroups, 0, 1, true);


}





/// This is it, man.  Do the conv outer-product.  Yee-hah.
extern "C" void convOutp(
		       cudamat* images,
		       cudamat* filters,
		       cudamat* targets,

		       int numModulesX,
		       int filterSize,
		       int paddingStart,
		       int moduleStride,
		       int numImgColors,
		       int numGroups,
			 int partialSum){

  _weightActsCu(images,
	      filters,
	      targets,

	      numModulesX,
	      filterSize,
	      paddingStart,
	      moduleStride,
	      numImgColors,
	      numGroups,
		  partialSum, 0, 1);


}
// Once this is done, I also need to implement pooling. Then I'll really
// be good. 






////////////////////////////////////////////////////////////////
/// POOLING

extern "C" void MaxPool(
			cudamat* images,
			cudamat* targets,
			int numFilters,
			int subsX,
			int startX,
			int strideX,
			int outputsX
			){

  // It features lots of good things in it. 
    MaxPooler mpooler;
    convLocalPoolCu<MaxPooler>(images, targets, 
                 numFilters,
			     subsX,
			     startX,
			     strideX,
			     outputsX,
			     mpooler);


}

extern "C" void AvgPool(
			cudamat* images,
			cudamat* targets,
			int numFilters,
			int subsX,
			int startX,
			int strideX,
			int outputsX,
			float div){
  // It features lots of good things in it. 
  AvgPooler apooler(div);
  convLocalPoolCu<AvgPooler>(images, targets,
			     numFilters,
			     subsX,
			     startX,
			     strideX,
			     outputsX,
			     apooler);

}

			
//void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
//                      int subsX, int startX, int strideX, int outputsX);


extern "C" void MaxPoolUndo(
			cudamat* images,
			cudamat* maxGrads,
			cudamat* maxActs, 
			cudamat* targets,

			int subsX,
			int startX,
			int strideX,
			int outputsX
			){
  convLocalMaxUndoCu(images, maxGrads, maxActs, targets, subsX, startX, strideX, outputsX, 0, 1);

}

extern "C" void AvgPoolUndo(
			cudamat* avgGrads,
			cudamat* targets,

			int subsX,
			int startX,
			int strideX,
			int outputsX,
            int imgSize
			){
  convLocalAvgUndoCu(avgGrads, targets, subsX, startX, strideX, outputsX, imgSize, 0, 1);

}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

extern "C" void localUp(
		       cudamat* images,
		       cudamat* filters,
		       cudamat* targets,
		       int numModulesX,
		       int paddingStart,
		       int moduleStride,
		       int numImgColors, 
		       int numGroups
		       //float scaleTargets,
		       //float scaleOutput
		       ){
  // Taken from _filterActs:

  // by default, scaleTargets = 0 and scaleOutput = 1
  int imagesStride = -1; //images->size[0];
  NVMatrix nvImages (images->data_device, 
		     images->size[0], 
		     images->size[1], 
		     imagesStride, 
		     false);

  //
  int filtersStride = -1; //filters->size[0];
  NVMatrix nvFilters(filters->data_device, 
		     filters->size[0], 
		     filters->size[1], 
		     filtersStride, 
		     false);

  // Why isn't it good? 
  int targetsStride = -1; //targets->size[0];

  // This way we copy the targets.  This is so fucking weird.
  NVMatrix nvTargets(targets->data_device, 
		     targets->size[0], 
		     targets->size[1], 
		     targetsStride, 
		     false);


  /* // These are actually for me. ConvFilterActs doesn't need it. */
  /* int numFilterColors = numImgColors / numGroups;       */
  /* int numFilters = nvFilters.getNumCols(); */
  /* int numModules = numModulesX * numModulesX; */
  /* int numImages = nvImages.getNumCols(); */
  /* int imgPixels = nvImages.getNumRows()/numImgColors; */
  /* int imgSize = int(sqrt(imgPixels)); */
  /* bool conv = 1; */
  /* int filterModuleMult = conv ? 1 : numModules; */


  //void convFilterActs
  /* NVMatrix& images,  */
  /*   NVMatrix& filters,  */
  /*   NVMatrix& targets, */
  /*   int numModulesX,  */
  /*   int paddingStart,  */
  /*   int moduleStride, */
  /*   int numImgColors,  */
  /*   int numGroups */


  // next, call it.
  localFilterActs(nvImages,
  		 nvFilters,
  		 nvTargets,
  		 numModulesX,
  		 paddingStart,
  		 moduleStride,
  		 numImgColors,
  		 numGroups) ;
    //scaleTargets,
    //scaleOutput);

}




extern "C" void localDown(
		       cudamat* images,
		       cudamat* filters,
		       cudamat* targets,


		       int imgSize,
		       int paddingStart,
		       int moduleStride,
		       int numImgColors,
		       int numGroups){

  int imagesStride = -1; //images->size[0];
  NVMatrix nvImages (images->data_device, 
		     images->size[0], 
		     images->size[1], 
		     imagesStride, 
		     false);

  //
  int filtersStride = -1; //filters->size[0];
  NVMatrix nvFilters(filters->data_device, 
		     filters->size[0], 
		     filters->size[1], 
		     filtersStride, 
		     false);

  // Why isn't it good? 
  int targetsStride = -1; //targets->size[0];

  // This way we copy the targets.  This is so fucking weird.
  NVMatrix nvTargets(targets->data_device, 
		     targets->size[0], 
		     targets->size[1], 
		     targetsStride, 
		     false);

  localImgActs(nvImages,
	      nvFilters,
	      nvTargets,

	      imgSize,
	      paddingStart,
	      moduleStride,
	      numImgColors,
	      numGroups);

}





/// This is it, man.  Do the conv outer-product.  Yee-hah.
extern "C" void localOutp(
		       cudamat* images,
		       cudamat* filters,
		       cudamat* targets,

		       int numModulesX,
		       int filterSize,
		       int paddingStart,
		       int moduleStride,
		       int numImgColors,
		       int numGroups
			 ){

  //NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
  //                       int numModulesX, int filterSize, int paddingStart, int moduleS//tride, int numImgColors, int numGroups, int partialSum)

  int imagesStride = -1; //images->size[0];
  NVMatrix nvImages (images->data_device, 
		     images->size[0], 
		     images->size[1], 
		     imagesStride, 
		     false);

  //
  int filtersStride = -1; //filters->size[0];
  NVMatrix nvFilters(filters->data_device, 
		     filters->size[0], 
		     filters->size[1], 
		     filtersStride, 
		     false);

  // Why isn't it good? 
  int targetsStride = -1; //targets->size[0];

  // This way we copy the targets.  This is so fucking weird.
  NVMatrix nvTargets(targets->data_device, 
		     targets->size[0], 
		     targets->size[1], 
		     targetsStride, 
		     false);

  localWeightActs(nvImages,
		  nvFilters,
		  nvTargets,

		  numModulesX,
		  filterSize,
		  paddingStart,
		  moduleStride,
		  numImgColors,
		  numGroups
		  );

}
// Once this is done, I also need to implement pooling. Then I'll really
// be good. 
