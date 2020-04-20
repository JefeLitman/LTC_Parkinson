"""Graph Functions for CNN3D to find the points on the video that
influence more in the classification. All the methods work with padding 
valid only.
Version: 5
Made by: Edgar Rangel and Luis Guayacan
"""
import numpy as np
import cv2

def index_max_full_connected(prev_index,W,X, n_importance = None):
    """Function that return the max index of a full connected
        Args:
            prev_index: Previous index of the maximum value.
            W: Weights of the layer to find the maximum value.  Its shape must be
                (input_neurons, output_neurons)
            X: Input of the layer. Its shape must be (Batch, input_neurons)
            n_importance: Positive integer that specify what index must be calculated by its importance.
                        Default value None and a value of 0 specify the most important.
    """
    XW = X*W[:,prev_index]
    if n_importance != None:
        if n_importance != 0:
            if n_importance < 0:
                raise ValueError('n_importance must be a positive integer, value passed: '+str(n_importance))
            else:
                for i in range(n_importance):
                    XW[0, np.argmax(XW)] = 0
    return np.argmax(XW)

#Serial function
def index_max_maxunpooling(prev_indexes,X,pool_size,stride):
    """Function that return the indexes of maximum values of maxpooling
    operation.
        Args:
        prev_indexes: Python list with tuples of indexes in the form 
                    (batch, frame, height, width, channels).
        X: Input of the maxpooling operation (Its shape must be before the maxpooling).
        pool_size: Size of the pool, must be a tuple of 3 elements (t,h,w)
        stride: How many points the pool move, must be a tuple of 3 elements (t,h,w)
    """
    max_indexes = []
    for pos in prev_indexes:
        position = np.r_[
            np.unravel_index(
                np.argmax(
                    X[pos[0],
                        pos[1]*stride[0] : pos[1]*stride[0] + pool_size[0],
                        pos[2]*stride[1] : pos[2]*stride[1] + pool_size[1],
                        pos[3]*stride[2] : pos[3]*stride[2] + pool_size[2],
                        pos[4]
                    ]
                ), pool_size
            )
        ] + [pos[1]*stride[0], pos[2]*stride[1], pos[3]*stride[2]]
        max_indexes.append( tuple( [pos[0]] + position.tolist() + [pos[4]] ) )
    return max_indexes

#Serial function
def index_max_deconvolution(prev_indexes, X, kernel, kernel_size, stride):
    """Function that return the indexes of maximum values of convolution
    operation.
        Args:
        prev_indexes: Python list with tuples of indexes in the form 
                    (batch, frame, height, width, channels).
        X: Input of the convolution operation. (Its shape must be before the convolution).
        kernel: Weights of the filter of convolution. Its shape must be 
                (t,h,w,input_filters, output_filters)
        kernel_size: Size of the kernel, must be a tuple of 3 elements (t,h,w)
        stride: How many points the kernel move, must be a tuple of 3 elements (t,h,w)
    """
    max_indexes = []
    for pos in prev_indexes:
        kernels = kernel[:,:,:,:,pos[4]]
        for c in range(kernels.shape[3]):
            position = np.r_[
                np.unravel_index(
                    np.argmax(
                        X[pos[0],
                            pos[1]*stride[0]:pos[1]*stride[0]+kernel_size[0],
                            pos[2]*stride[1]:pos[2]*stride[1]+kernel_size[1],
                            pos[3]*stride[2]:pos[3]*stride[2]+kernel_size[2],
                            c
                        ] * kernels[:,:,:,c]
                    ), kernel_size
                )
            ] + [pos[1]*stride[0], pos[2]*stride[1], pos[3]*stride[2]]
            max_indexes.append( tuple( [pos[0]] + position.tolist() + [c]) )
    return max_indexes

#Parallel function
def index_max_maxunpooling_parallel(args):
    """Function that return the indexes of maximum values of maxpooling
    operation.
        Args:
        args: Tuple of 4 elements with the next inside:
                - prev_index: Python tuple of index in the form (batch, frame, height, width, channels).
                - X: Input of the maxpooling operation (Its shape must be before the maxpooling).
                - pool_size: Size of the pool, must be a tuple of 3 elements (t,h,w)
                - stride: How many points the pool move, must be a tuple of 3 elements (t,h,w)
    """
    position = np.r_[
        np.unravel_index(
            np.argmax(
                args[1]
                [   
                    args[0][0],
                    args[0][1]*args[3][0] : args[0][1]*args[3][0] + args[2][0],
                    args[0][2]*args[3][1] : args[0][2]*args[3][1] + args[2][1],
                    args[0][3]*args[3][2] : args[0][3]*args[3][2] + args[2][2],
                    args[0][4]
                ]
            ), args[2]
        )
    ] + [args[0][1]*args[3][0], args[0][2]*args[3][1], args[0][3]*args[3][2]]
    return tuple( [args[0][0]] + position.tolist() + [args[0][4]] )

#Parallel function
def index_max_deconvolution_parallel(args):
    """Function that return the indexes of maximum values of convolution
    operation.
        Args:
        args: Tuple of 5 elements with the next inside:
            - prev_index: Python tuple of index in the form (batch, frame, height, width, channels).
            - X: Input of the convolution operation. (Its shape must be before the convolution).
            - kernel: Weights of the filter of convolution. Its shape must be (t,h,w,input_filters)
            - kernel_size: Size of the kernel, must be a tuple of 3 elements (t,h,w)
            - stride: How many points the kernel move, must be a tuple of 3 elements (t,h,w)
    """
    max_indexes = []
    for c in range(args[2].shape[3]):
        position = np.r_[
            np.unravel_index(
                np.argmax(
                    args[1]
                    [
                        args[0][0],
                        args[0][1]*args[4][0] : args[0][1]*args[4][0] + args[3][0],
                        args[0][2]*args[4][1] : args[0][2]*args[4][1] + args[3][1],
                        args[0][3]*args[4][2] : args[0][3]*args[4][2] + args[3][2],
                        c
                    ] * args[2][:,:,:,c]
                ), args[3]
            )
        ] + [args[0][1]*args[4][0], args[0][2]*args[4][1], args[0][3]*args[4][2]]
        max_indexes.append( tuple( [args[0][0]] + position.tolist() + [c] ) )
    return max_indexes

#Serial function
def graph_indexes_on_video(video, indexes, color_map ,importance, n_max):
    """Function that return the video with the important points painted in it. It takes the 
    all the frames and paint circles with the color by color map.
    operation.
        Args:
        video: Numpy array with values between 0 and 1 in the form (frames, height, width, channels).
        indexes: Numpy array the indexes to paint. Its shape must be (n_points, 5) where the 5 is 
                (batch, frame, height, width, channels).
        color_map: matplotlib color map selected with the range of color to paint.
        importance: Integer between 0 and the range of the color map to paint the points.
        n_max : Integer that represents the number of maximum points to graph, generally is the len
                of the color_map.
    """
    raw_video = video.copy()

    for frame in range(video.shape[0]):
        if frame in np.unique(indexes[:,1]):
            for index in np.argwhere(indexes[:,1] == frame):
                raw_video[frame,indexes[index,2], indexes[index,3],:] += np.r_[color_map(importance/n_max)[-2::-1]]
                video[frame] = cv2.addWeighted(raw_video[frame], 0.0008, video[frame], 1 - 0.0008, 0)
        video[frame] = cv2.cvtColor(video[frame],cv2.COLOR_BGR2RGB)
    return video

#Parallel function
def graph_indexes_on_video_parallel(args):
    """Function that return the video with the important points painted in it. It takes the 
    all the frames and paint circles with the color by color map.
    operation.
        Args:
        args: Tuple of 6 elements with the next inside:
            - frame: Numpy array with values between 0 and 1 in the form (height, width, channels).
            - frame_index: Integer that specify in what frame of the video is painting the circles. From 0 to 
                    number of frames.
            - indexes: Numpy array the indexes to paint. Its shape must be (n_points, 5) where the 5 is 
                    (batch, frame, height, width, channels).
            - color_map: matplotlib color map selected with the range of color to paint.
            - importance: Integer between 0 and the range of the color map to paint the points.
            - n_max : Integer that represents the number of maximum points to graph, generally is the len
                    of the color_map.
    """
    frame = args[0]
    raw_frame = frame.copy()
    if args[1] in np.unique(args[2][:,1]):
        for index in np.argwhere(args[2][:,1] == args[1]):
            raw_frame[args[2][index,2], args[2][index,3],:] = np.r_[args[3](args[4]/args[5])[-2::-1]]
            frame = cv2.addWeighted(raw_frame, 0.0008, frame, 1 - 0.0008, 0)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    return frame