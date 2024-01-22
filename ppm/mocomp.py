import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
NOTES: using (CORRECTED) code from skvideo
from skvideo.motion import blockMotion, blockComp

logic:
for each block in target find closest match in anchor
reccord the displacement
predicted blocks are cuts from anchor according to displacement

    INCORRECT
    compVid[i, :, :, :] = _subcomp(videodata[i], motionVect[i-1], mbSize)
    Proposed solution:
    compVid[i, :, :, :] = _subcomp(videodata[i-1], motionVect[i-1], mbSize)

there is a y x reversal lurking in DS, ARPS 
but not in ES...
"""

def _costMAD(block1, block2):
    block1 = block1.astype(np.float32)
    block2 = block2.astype(np.float32)
    return np.mean(np.abs(block1 - block2))

def _minCost(costs):
    h, w = costs.shape
    mi = costs[int((h-1)/2), int((w-1)/2)]
    dy = int((h-1)/2)
    dx = int((w-1)/2)
    #mi = 65535
    #dy = 0
    #dx = 0

    for i in range(h): 
      for j in range(w): 
        if costs[i, j] < mi:
          mi = costs[i, j]
          dy = i
          dx = j

    return dx, dy, mi

def _checkBounded(xval, yval, w, h, mbSize):
    if ((yval < 0) or
       (yval + mbSize >= h) or
       (xval < 0) or
       (xval + mbSize >= w)):
        return False
    else:
        return True


def _DS(imgP, imgI, mbSize, p):
    # Computes motion vectors using Diamond Search method
    #
    # Input
    #   imgP : The image for which we want to find motion vectors
    #   imgI : The reference image
    #   mbSize : Size of the macroblock
    #   p : Search parameter  (read literature to find what this means)
    #
    # Ouput
    #   motionVect : the motion vectors for each integral macroblock in imgP
    #   DScomputations: The average number of points searched for a macroblock

    h, w = imgP.shape

    vectors = np.zeros((int(h / mbSize), int(w / mbSize), 2))
    costs = np.ones((9))*65537

    L = np.floor(np.log2(p + 1))

    LDSP = []
    LDSP.append([0, -2])
    LDSP.append([-1, -1])
    LDSP.append([1, -1])
    LDSP.append([-2, 0])
    LDSP.append([0, 0])
    LDSP.append([2, 0])
    LDSP.append([-1, 1])
    LDSP.append([1, 1])
    LDSP.append([0, 2])

    SDSP = []
    SDSP.append([0, -1])
    SDSP.append([-1, 0])
    SDSP.append([0, 0])
    SDSP.append([1, 0])
    SDSP.append([0, 1])

    computations = 0

    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i
            costs[4] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])
            cost = 0
            point = 4
            if costs[4] != 0:
                computations += 1
                for k in range(9):
                    refBlkVer = y + LDSP[k][1]
                    refBlkHor = x + LDSP[k][0]
                    if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue
                    if k == 4:
                        continue
                    costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    computations += 1

                point = np.argmin(costs)
                cost = costs[point]

            SDSPFlag = 1
            if point != 4:
                SDSPFlag = 0
                cornerFlag = 1
                if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):
                    cornerFlag = 0
                xLast = x
                yLast = y
                x = x + LDSP[point][0]
                y = y + LDSP[point][1]
                costs[:] = 65537
                costs[4] = cost

            while SDSPFlag == 0:
                if cornerFlag == 1:
                    for k in range(9):
                        refBlkVer = y + LDSP[k][1]
                        refBlkHor = x + LDSP[k][0]
                        if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        if k == 4:
                            continue

                        if ((refBlkHor >= xLast - 1) and
                           (refBlkHor <= xLast + 1) and
                           (refBlkVer >= yLast - 1) and
                           (refBlkVer <= yLast + 1)):
                            continue
                        elif ((refBlkHor < j-p) or
                              (refBlkHor > j+p) or
                              (refBlkVer < i-p) or
                              (refBlkVer > i+p)):
                            continue
                        else:
                            costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                            computations += 1
                else:
                    lst = []
                    if point == 1:
                        lst = np.array([0, 1, 3])
                    elif point == 2:
                        lst = np.array([0, 2, 5])
                    elif point == 6:
                        lst = np.array([3, 6, 8])
                    elif point == 7:
                        lst = np.array([5, 7, 8])

                    for idx in lst:
                        refBlkVer = y + LDSP[idx][1]
                        refBlkHor = x + LDSP[idx][0]
                        if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                            continue
                        elif ((refBlkHor < j - p) or
                              (refBlkHor > j + p) or
                              (refBlkVer < i - p) or
                              (refBlkVer > i + p)):
                            continue
                        else:
                            costs[idx] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                            computations += 1

                point = np.argmin(costs)
                cost = costs[point]

                SDSPFlag = 1
                if point != 4:
                    SDSPFlag = 0
                    cornerFlag = 1
                    if (np.abs(LDSP[point][0]) == np.abs(LDSP[point][1])):
                        cornerFlag = 0
                    xLast = x
                    yLast = y
                    x += LDSP[point][0]
                    y += LDSP[point][1]
                    costs[:] = 65537
                    costs[4] = cost
            costs[:] = 65537
            costs[2] = cost

            for k in range(5):
                refBlkVer = y + SDSP[k][1]
                refBlkHor = x + SDSP[k][0]

                if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                    continue
                elif ((refBlkHor < j - p) or
                      (refBlkHor > j + p) or
                      (refBlkVer < i - p) or
                      (refBlkVer > i + p)):
                    continue

                if k == 2:
                    continue

                costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                computations += 1

            point = 2
            cost = 0 
            if costs[2] != 0:
                point = np.argmin(costs)
                cost = costs[point]

            x += SDSP[point][0]
            y += SDSP[point][1]

            vectors[int(i / mbSize), int(j / mbSize), :] = [x - j, y - i]

            costs[:] = 65537

    return vectors, computations / ((h * w) / mbSize**2)

# Exhaustive Search
def _ES(imgP, imgI, mbSize, p):
    h, w = imgP.shape

    vectors = np.zeros((int(h / mbSize), int(w / mbSize), 2), dtype=np.float32)
    costs = np.ones((2 * p + 1, 2 * p + 1), dtype=np.float32)*65537

    # we start off from the top left of the image
    # we will walk in steps of mbSize
    # for every marcoblock that we look at we will look for
    # a close match p pixels on the left, right, top and bottom of it
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            # the exhaustive search starts here
            # we will evaluate cost for  (2p + 1) blocks vertically
            # and (2p + 1) blocks horizontaly
            # m is row(vertical) index
            # n is col(horizontal) index
            # this means we are scanning in raster order

            if ((j + p + mbSize >= w) or
                (j - p < 0) or
                (i - p < 0) or
                (i + p + mbSize >= h)):
                for m in range(-p, p + 1):
                    for n in range(-p, p + 1):
                        refBlkVer = i + m   # row/Vert co-ordinate for ref block
                        refBlkHor = j + n   # col/Horizontal co-ordinate
                        if ((refBlkVer < 0) or
                           (refBlkVer + mbSize > h) or
                           (refBlkHor < 0) or
                           (refBlkHor + mbSize > w)):
                                continue

                        costs[m + p, n + p] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])

            else:
                for m in range(-p, p + 1):
                    for n in range(-p, p + 1):
                        refBlkVer = i + m   # row/Vert co-ordinate for ref block
                        refBlkHor = j + n   # col/Horizontal co-ordinate
                        costs[m + p, n + p] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])


            # Now we find the vector where the cost is minimum
            # and store it ... this is what will be passed back.
            dx, dy, mi = _minCost(costs)  # finds which macroblock in imgI gave us min Cost
            vectors[int(i / mbSize), int(j / mbSize), :] = [dy - p, dx - p]

            costs[:, :] = 65537

    return vectors

def blockMotion(videodata, method='DS', mbSize=8, p=2, **plugin_args):
    """Block-based motion estimation
    
    Given a sequence of frames, this function
    returns motion vectors between frames.

    Parameters
    ----------
    videodata : ndarray, shape (numFrames, height, width, channel)
        A sequence of frames

    method : string
        "ES" --> exhaustive search

        "3SS" --> 3-step search

        "N3SS" --> "new" 3-step search [#f1]_

        "SE3SS" --> Simple and Efficient 3SS [#f2]_

        "4SS" --> 4-step search [#f3]_

        "ARPS" --> Adaptive Rood Pattern search [#f4]_

        "DS" --> Diamond search [#f5]_

    mbSize : int
        Macroblock size

    p : int
        Algorithm search distance parameter

    Returns
    ----------
    motionData : ndarray, shape (numFrames - 1, height/mbSize, width/mbSize, 2)

        The motion vectors computed from videodata. The first element of the last axis contains the y motion component, and second element contains the x motion component.

    References
    ----------
    .. [#f1] Renxiang Li, Bing Zeng, and Ming L. Liou, "A new three-step search algorithm for block motion estimation." IEEE Transactions on Circuits and Systems for Video Technology, 4 (4) 438-442, Aug 1994

    .. [#f2] Jianhua Lu and Ming L. Liou, "A simple and efficient search algorithm for block-matching motion estimation." IEEE Transactions on Circuits and Systems for Video Technology, 7 (2) 429-433, Apr 1997

    .. [#f3] Lai-Man Po and Wing-Chung Ma, "A novel four-step search algorithm for fast block motion estimation." IEEE Transactions on Circuits and Systems for Video Technology, 6 (3) 313-317, Jun 1996

    .. [#f4] Yao Nie and Kai-Kuang Ma, "Adaptive rood pattern search for fast block-matching motion estimation." IEEE Transactions on Image Processing, 11 (12) 1442-1448, Dec 2002

    .. [#f5] Shan Zhu and Kai-Kuang Ma, "A new diamond search algorithm for fast block-matching motion estimation." IEEE Transactions on Image Processing, 9 (2) 287-290, Feb 2000

    """
    # videodata = vshape(videodata)
    T, H, W, C = videodata.shape
    # grayscale
    if C == 1:
        luminancedata = videodata
    elif C == 3: # assume RGB
        luminancedata = videodata[:, :, :, 0]*0.2989 + videodata[:, :, :, 1]*0.5870 + videodata[:, :, :, 2]*0.1140 

    numFrames, height, width, channels = luminancedata.shape
    assert numFrames > 1, "Must have more than 1 frame for motion estimation!"

    # luminance is 1 channel, so flatten for computation
    luminancedata = luminancedata.reshape((numFrames, height, width))

    motionData = np.zeros((numFrames - 1, int(height / mbSize), int(width / mbSize), 2), np.int8)

    if method == "ES":
        for i in range(numFrames - 1):
            motion = _ES(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    elif method == "DS":
        for i in range(numFrames - 1):
            motion, comps = _DS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    else:
        raise NotImplementedError

    return motionData

#only handles (M, N, C) shapes
def _subcomp(framedata, motionVect, mbSize):
    M, N, C = framedata.shape

    compImg = np.zeros((M, N, C))

    for i in range(0, M - mbSize + 1, mbSize):
        for j in range(0, N - mbSize + 1, mbSize):
            dy = motionVect[int(i / mbSize), int(j / mbSize), 0]
            dx = motionVect[int(i / mbSize), int(j / mbSize), 1]

            refBlkVer = i + dy
            refBlkHor = j + dx

            # check bounds
            if not _checkBounded(refBlkHor, refBlkVer, N, M, mbSize):
                continue

            compImg[i:i + mbSize, j:j + mbSize, :] = framedata[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize, :]
    return compImg

def blockComp(videodata, motionVect, mbSize=8):
    """Block-based motion compensation
    
    Using the given motion vectors, this function
    returns the motion-compensated video data.

    Parameters
    ----------
    videodata : ndarray
        an input frame sequence, shape (T, M, N, C), (T, M, N), (M, N, C) or (M, N)

    motionVect : ndarray
        ndarray representing block motion vectors. Expects ndarray, shape (T-1, M/mbSize, N/mbSize) or (M/mbSize, N/mbSize).

    mbSize : int
        Size of macroblock in pixels.

    Returns
    -------
    compImg : ndarray
        ndarray holding the motion compensated image frame, shape (T, M, N, C)

    """

    # videodata = vshape(videodata)
    T, M, N, C = videodata.shape

    if T == 1:	# a single frame is passed in
        return _subcomp(videodata, motionVect, mbSize)

    else: # more frames passed in
        # allocate compensation data
        compVid = np.zeros((T, M, N, C))
        # pass the first frame uncorrected
        compVid[0, :, :, :] = videodata[0]
        for i in range(1, T):
            compVid[i, :, :, :] = _subcomp(videodata[i-1], motionVect[i-1], mbSize)
        return compVid

def gauss(cx, cy, sigma, sz):
    data = np.zeros((sz, sz, 1), dtype=np.float32)
    for y in range(-sigma*3, sigma*3+1, 1):
        for x in range(-sigma*3, sigma*3+1, 1):
            magnitude = np.exp(-0.5 * (x**2 + y**2)/(sigma**2))
            if ((x + cx < 0) or (x + cx > sz) or (y + cy < 0) or (y + cy > sz)):
                continue
            data[y + cy, x + cx, 0] = magnitude
            # data[y + cy, x + cx, 1] = magnitude
            # data[y + cy, x + cx, 2] = magnitude
    return data

def get_blob_displacement():
    frame1 = gauss(50, 50, 5, 128)
    frame2 = gauss(55, 50, 5, 128)
    frame3 = np.zeros(frame2.shape)
    videodata = []
    videodata.append(frame1)
    videodata.append(frame2)
    videodata.append(frame3)
    videodata = np.array(videodata)
    return videodata

def get_rigid_motion(S):
    import os
    home = os.path.expanduser("~")
    f = home + "/Documents/datasets/DAVIS/JPEGImages/480p/bear/" 
    img0 = np.array(Image.open(f + "00000.jpg").convert('L'))[None, ..., None]
    videodata = img0 / 255.
    H, W = videodata.shape[1:3]
    videodata = videodata[:, H//2-S:H//2+S, W//2-S:W//2+S]
    for t in range(2):
        videodata = np.concatenate((videodata, np.roll(videodata[-1:], shift=5, axis=2)))
    for t in range(2):
        videodata = np.concatenate((videodata, np.roll(videodata[-1:], shift=5, axis=1)))
    return videodata

def get_natural_motion(S):
    import os
    home = os.path.expanduser("~")
    f = home + "/Documents/datasets/DAVIS/JPEGImages/480p/bear/" 
    img0 = np.array(Image.open(f + "00000.jpg").convert('L'))[None, ..., None]
    img1 = np.array(Image.open(f + "00001.jpg").convert('L'))[None, ..., None]
    img2 = np.array(Image.open(f + "00002.jpg").convert('L'))[None, ..., None]
    img3 = np.array(Image.open(f + "00003.jpg").convert('L'))[None, ..., None]
    videodata = np.concatenate([img0, img1, img2], axis=0) / 255.
    H, W = videodata.shape[1:3]
    videodata = videodata[:, H//2-S:H//2+S, W//2-S:W//2+S]
    return videodata

def flip(motion):
    motion_flip = np.zeros_like(motion)
    motion_flip[..., 0] = motion[..., 1]
    motion_flip[..., 1] = motion[..., 0]
    return motion_flip

def display_image_and_flow(videodata, motion, B=8, t=0, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        Vy = motion[t, ..., 0]
        Vx = motion[t, ..., 1]
        img = videodata[t+1, ..., 0]
        ydim, xdim = img.shape
        ax.imshow(img, cmap='gray')
        X, Y = np.meshgrid(np.arange(0, xdim - B + 1, B) + B//2,
                           np.arange(0, ydim - B + 1, B) + B//2)
        Vy = -1*Vy
        assert X.shape == Vx.shape, f"{X.shape}, {Vx.shape}"
        _ = ax.quiver(X, Y, Vx, Vy, scale=50, color='r',
                       alpha=.8, width=.005, minlength=0)
        plt.axis('off')
        return fig

def debug():
    import plenoptic as po
    B = 8
    P = 8
    tau = 1
    for videodata in [get_blob_displacement(), 
                      get_rigid_motion(S=64),
                      get_natural_motion(S=64)]:

        po.imshow(videodata[None, ..., 0], vrange='auto1', title=None); 

        copy_residual = np.diff(videodata, axis=0)[:, B:-B, B:-B, 0]
        po.imshow(copy_residual[:5][None, ...], vrange='auto0', title=None); 

        motion = blockMotion(videodata, method="DS", mbSize=B, p=P)
        motion = flip(motion)

        plt.figure()
        plt.title('distribution of motion')
        plt.hist(motion.flatten(), bins=2*P+1);
        plt.show()

        display_image_and_flow(videodata, motion, B=8, t=0)
        display_image_and_flow(videodata, motion, B=8, t=1)

        compmotion = blockComp(videodata, motion, mbSize=B)
        residual = (videodata - compmotion)[1:, B:-B, B:-B, 0]
        po.imshow(residual[:5][None, ...], vrange='auto0', title=None);

        po.imshow(compmotion[1:, B:-B, B:-B][:5, :, :, 0][None, ...], vrange='auto1', title=None); 

        tau = 1
        causal_compmotion = blockComp(videodata[tau:], motion[:-tau], mbSize=B)
        causal_residual = (videodata[tau:] - causal_compmotion)[1:, B:-B, B:-B, 0]
        po.imshow(causal_residual[:5][None, ...], vrange='auto0', title=None);

        C_RMSE = (copy_residual ** 2).mean((-2, -1)) ** .5
        MC_RMSE = (residual ** 2).mean((-2, -1)) ** .5
        cMC_RMSE = (causal_residual ** 2).mean((-2, -1)) ** .5
        plt.figure()
        plt.plot(np.arange(1, len(C_RMSE)+1), C_RMSE, label="copy")
        plt.plot(np.arange(1, len(MC_RMSE)+1), MC_RMSE, label="mocomp")
        plt.plot(np.arange(2, len(cMC_RMSE)+2), cMC_RMSE, 'o', label="causal mocomp")
        plt.legend()
        plt.ylabel("RMSE")
        plt.xlabel("time")
        plt.show()

        plt.figure()
        plt.bar(['C', 'MC', 'cMC'], [C_RMSE.mean(), MC_RMSE.mean(), cMC_RMSE.mean()])
        plt.ylabel("mean RMSE")
        plt.show()
