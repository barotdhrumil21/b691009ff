import os
import json
import numpy as np
import cv2
from time import localtime,strftime,time
import pandas as pd
from csv import QUOTE_NONNUMERIC
import re
import glob
import sys
import string
import random
import pdfkit

#imports from globals.py
OMR_INPUT_DIR ='./functions/inputs/OMR_Files/'
saveMarkedDir='./functions/outputs/CheckedOMRs/'
resultDir='./functions/outputs/Results/'
manualDir='./functions/outputs/Manual/'
errorsDir=manualDir+'ErrorFiles/'
badRollsDir=manualDir+'BadRollNosFiles/'
multiMarkedDir=manualDir+'MultiMarkedFiles/'
ERODE_SUB_OFF = 1
saveMarked = 1
showimglvl = 2
saveimglvl = 0
PRELIM_CHECKS=0
saveImgList = {}
resetpos = [0,0]
explain= 0
TEXT_SIZE=0.95
CLR_BLACK = (50,150,150)
CLR_WHITE = (250,250,250)
CLR_GRAY = (120,120,120)
# CLR_DARK_GRAY = (190,190,190)
CLR_DARK_GRAY = (90,90,90)
uniform_height = int(1231 / 1.5)
uniform_width = int(1000 / 1.5)
MIN_GAP, MIN_STD = 30, 25
MIN_JUMP = 18
MIN_PAGE_AREA = 80000
# If only not confident, take help of globalTHR
CONFIDENT_JUMP = MIN_JUMP+15
JUMP_DELTA = 30
# MIN_GAP : worst case gap of black and gray
display_height = int(480)
display_width  = int(640)
windowWidth = 1280
windowHeight = 720
filesMoved=0
filesNotMoved=0

# for positioning image windows
windowX,windowY = 0,0

Answers={
        'q1': ['A'],'q2': ['A'],'q3': ['A'],'q4': ['A'],'q5': ['A'],'q6': ['A'],'q7': ['A'],
        'q8': ['A'],'q9': ['A'],'q10': ['A'],'q11': ['A'],'q12': ['A'],'q13': ['A'],'q14': ['A'],
        'q15': ['A'],'q16': ['A'],'q17': ['A'],'q18': ['A'],'q19': ['A'],'q20': ['A'],'q21': ['A'],
        'q22': ['A'],'q23': ['A'],'q24': ['A'],'q25': ['A'],'q26': ['A'],'q27': ['A'],'q28': ['A'],
        'q29': ['A'],'q30': ['A'],'q31': ['A'],'q32': ['A'],'q33': ['A'],'q34': ['A'],'q35': ['A'],
        'q36': ['A'],'q37': ['A'],'q38': ['A'],'q39': ['A'],'q40': ['A'],'q41': ['A'],'q42': ['A'],
        'q43': ['A'],'q44': ['A'],'q45': ['A'],'q46': ['A'],'q47': ['A'],'q48': ['A'],'q49': ['A'],
        'q50': ['A'],'q51': ['A'],'q52': ['A'],'q53': ['A'],'q54': ['A'],'q55': ['A'],'q56': ['A'],
        'q57': ['A'],'q58': ['A'],'q59': ['A'],'q60': ['A'],'q61': ['A'],'q62': ['A'],'q63': ['A'],
        'q64': ['A'],'q65': ['A'],'q66': ['A'],'q67': ['A'],'q68': ['A'],'q69': ['A'],'q70': ['A'],
        'q71': ['A'],'q72': ['A'],'q73': ['A'],'q74': ['A'],'q75': ['A'],'q76': ['A'],'q77': ['A'],
        'q78': ['A'],'q79': ['A'],'q80': ['A'],'q81': ['A'],'q82': ['A'],'q83': ['A'],'q84': ['A'],
        'q85': ['A'],'q86': ['A'],'q87': ['A'],'q88': ['A'],'q89': ['A'],'q90': ['A'],'q91': ['A'],
        'q92': ['A'],'q93': ['A'],'q94': ['A'],'q95': ['A'],'q96': ['A'],'q97': ['A'],'q98': ['A'],
        'q99': ['A'],'q100': ['A'],
        }

Sections = {
    #"MM":5
    'Custom':{
        'ques':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
         44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
         67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
         90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
         'marks':1
         }

}
#end imports from globals.py

#imports from template.py

qtype_data = {
    'QTYPE_MED':{
        'vals' : ['E','H'],
        'orient':'V'
    },
    'QTYPE_ROLL':{
        'vals':range(10),
        'orient':'V'
    },
    'QTYPE_INT':{
        'vals':range(10),
        'orient':'V'
    },
    'QTYPE_MCQ4':{
        'vals' : ['A','B','C','D'],
        'orient':'H'
    },
    'QTYPE_MCQ5':{
        'vals' : ['A','B','C','D','E'],
        'orient':'H'
    },
    # Add custom question types here-
    # ,
    # 'QTYPE_MCQ_COL_5A':{'vals' : ['A']*5, 'orient':'V'},
    # 'QTYPE_MCQ_COL_5B':{'vals' : ['B']*5, 'orient':'V'},
    # 'QTYPE_MCQ_COL_5C':{'vals' : ['C']*5, 'orient':'V'},
    # 'QTYPE_MCQ_COL_5D':{'vals' : ['D']*5, 'orient':'V'},
    # 'QTYPE_MCQ_COL_4A':{'vals' : ['A']*4, 'orient':'V'},
    # 'QTYPE_MCQ_COL_4B':{'vals' : ['B']*4, 'orient':'V'},
    # 'QTYPE_MCQ_COL_4C':{'vals' : ['C']*4, 'orient':'V'},
    # 'QTYPE_MCQ_COL_4D':{'vals' : ['D']*4, 'orient':'V'},
}

class Pt():
    def __init__(self, pt, qNo, qType, val):
        self.x=pt[0]
        self.y=pt[1]
        self.qNo=qNo
        self.qType=qType
        self.val=val


class QBlock():
    def __init__(self, dims, key, orig, traverse_pts):
        # dims = (width, height)
        self.dims = dims
        self.key = key
        self.orig = orig
        self.traverse_pts = traverse_pts
        # will be set when using
        self.shift = 0


def genQBlock(bubbleDims, QBlockDims, key, orig, qNos, gaps, vals, qType, orient, col_orient):
    H, V = (0,1) if(orient=='H') else (1,0)

    Qs=[]
    traverse_pts = []
    o = orig.copy()
    if(col_orient == orient):
        for q in range(len(qNos)):
            pt = o.copy()
            pts = []
            for v in range(len(vals)):
                pts.append(Pt(pt.copy(),qNos[q],qType,vals[v]))
                pt[H] += gaps[H]
            # For diagonalal endpoint of QBlock
            pt[H] += bubbleDims[H] - gaps[H]
            pt[V] += bubbleDims[V]
            #TODO- make a mini object for this
            traverse_pts.append(([o.copy(), pt.copy()], pts))
            o[V] += gaps[V]
    else:
        for v in range(len(vals)):
            pt = o.copy()
            pts = []
            for q in range(len(qNos)):
                pts.append(Pt(pt.copy(),qNos[q],qType,vals[v]))
                pt[V] += gaps[V]
            # For diagonalal endpoint of QBlock
            pt[V] += bubbleDims[V] - gaps[V]
            pt[H] += bubbleDims[H]
            traverse_pts.append(([o.copy(), pt.copy()], pts))
            o[H] += gaps[H]
    return QBlock(QBlockDims, key, orig, traverse_pts)


def genGrid(bubbleDims, key, qType, orig, bigGaps, gaps, qNos, vals, orient='V', col_orient='V'):
    gridData = np.array(qNos)
    if(0 and len(gridData.shape)!=3 or gridData.size==0):
        exit(4)
        return []

    orig = np.array(orig)
    numQsMax = max([max([len(qb) for qb in row]) for row in gridData])

    numDims = [numQsMax, len(vals)]

    QBlocks=[]

    H, V = (0,1) if(orient=='H') else (1,0)

    qStart = orig.copy()

    origGap = [0, 0]

    # Usually single row
    for row in gridData:
        qStart[V] = orig[V]

        # Usually multiple qTuples
        for qTuple in row:
            # Update numDims and origGaps
            numDims[0] = len(qTuple)
            # bigGaps is indep of orientation
            origGap[0] = bigGaps[0] + (numDims[V]-1)*gaps[H]
            origGap[1] = bigGaps[1] + (numDims[H]-1)*gaps[V]
            # each qTuple will have qNos
            QBlockDims = [
                # width x height in pixels
                gaps[0] * (numDims[V]-1) + bubbleDims[H],
                gaps[1] * (numDims[H]-1) + bubbleDims[V]
            ]
            # WATCH FOR BLUNDER(use .copy()) - qStart was getting passed by reference! (others args read-only)
            QBlocks.append(genQBlock(bubbleDims, QBlockDims, key, qStart.copy(),qTuple,gaps,vals,qType,orient,col_orient))
            # Goes vertically down first
            qStart[V] += origGap[V]
        qStart[H] += origGap[H]
    return QBlocks


class Template():
    def __init__(self, jsonObj):
        self.QBlocks = []
        self.dims = jsonObj["Dimensions"]
        self.bubbleDims = jsonObj["BubbleDimensions"]
        self.concats = jsonObj["Concatenations"]
        self.singles = jsonObj["Singles"]

    def addQBlocks(self, key, rect):
        assert(self.bubbleDims != [-1, -1])
        self.QBlocks += genGrid(self.bubbleDims, key, **rect,**qtype_data[rect['qType']])


def read_template(filename):
    with open(filename, "r") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(e)
            exit(5)

templJSON={}
TEMPLATE_FILE = "./functions/inputs"+"/template.json";
if(os.path.exists(TEMPLATE_FILE)):
    templJSON = read_template(TEMPLATE_FILE)
else:
    print("template json does not exist", os.listdir('.'))
    print(os.path.exists('./functions/inputs/'),"-",os.getcwd())

if(len(templJSON.keys()) == 0):
    exit(6)

TEMPLATES={}

TEMPLATES = Template(templJSON)
for k, QBlocks in templJSON.items():
    if(k not in ["Dimensions","BubbleDimensions","Concatenations","Singles"]):
        # Add QBlock to array of grids
        TEMPLATES.addQBlocks(k, QBlocks)
#end imports from template.py

#imports from utils.py

for _dir in [saveMarkedDir]:
    if(not os.path.exists(_dir)):
        os.makedirs(_dir)
        os.mkdir(_dir+'/stack')
        os.mkdir(_dir+'/_MULTI_')
        os.mkdir(_dir+'/_MULTI_'+'/stack')
    else:
        pass

for _dir in [manualDir,resultDir]:
    if(not os.path.exists(_dir)):
            os.makedirs(_dir)
    else:
        pass

for _dir in [multiMarkedDir,errorsDir,badRollsDir]:
    if(not os.path.exists(_dir)):
        os.makedirs(_dir)
    else:
        pass

def saveImg(path, final_marked):
    cv2.imwrite(path,final_marked)

def waitQ():
    ESC_KEY = 27
    while(cv2.waitKey(1) & 0xFF not in [ord('q'), ESC_KEY]):pass
    cv2.destroyAllWindows()

#not sure why following 2 functions are here!
def resetSaveImg(key):
    global saveImgList
    saveImgList[key] = []

def appendSaveImg(key,img):
    if(saveimglvl >= int(key)):
        global saveImgList
        if(key not in saveImgList):
            saveImgList[key] = []
        saveImgList[key].append(img.copy())

def normalize_util(img, alpha=0, beta=255):
    return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)

def resize_util(img, u_width, u_height=None):
    if u_height == None:
        h,w=img.shape[:2]
        u_height = int(h*u_width/w)
    return cv2.resize(img,(u_width,u_height))

def resize_util_h(img, u_height, u_width=None):
    if u_width == None:
        h,w=img.shape[:2]
        u_width = int(w*u_height/h)
    return cv2.resize(img,(u_width,u_height))


def show(name,orig,pause=1,resize=False,resetpos=None):
    if 1 == 1:
        pass
    else:
        global windowX, windowY, display_width
        if(type(orig) == type(None)):
            if(pause):
                cv2.destroyAllWindows()
            return
        origDim = orig.shape[:2]
        img = resize_util(orig,display_width,display_height) if resize else orig
        cv2.imshow(name,img)
        if(resetpos):
            windowX=resetpos[0]
            windowY=resetpos[1]
        cv2.moveWindow(name,windowX,windowY)

        h,w = img.shape[:2]

        # Set next window position
        margin = 25
        w += margin
        h += margin
        if(windowX+w > windowWidth):
            windowX = 0
            if(windowY+h > windowHeight):
                windowY = 0
            else:
                windowY += h
        else:
            windowX += w

        if(pause):
            waitQ()

def saveOrShowStacks(key, name, savedir=None,pause=1):
    global saveImgList
    if(saveimglvl >= int(key) and saveImgList[key]!=[]):
        result = np.hstack(tuple([resize_util_h(img,uniform_height) for img in saveImgList[key]]))
        result = resize_util(result,min(len(saveImgList[key])*uniform_width//3,int(uniform_width*2.5)))
        if (type(savedir) != type(None)):
            saveImg(savedir+'stack/'+name+'_'+str(key)+'_stack.jpg', result)
        else:
            pass



clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

def getROI(image, filename = None, noCropping=False, noMarkers=True):
    global clahe, marker_eroded_sub
    for i in range(saveimglvl):
        resetSaveImg(i+1)

    appendSaveImg(1,image)
    img = image.copy()
    img = cv2.GaussianBlur(img,(3,3),0)
    image_norm = normalize_util(img);

    image_norm = resize_util(image_norm, uniform_width, uniform_height)
    image = resize_util(image, uniform_width, uniform_height)
    appendSaveImg(1,image_norm)

    if(noMarkers == True):
        return image_norm
    else:
        image_eroded_sub = normalize_util(image_norm) if ERODE_SUB_OFF else normalize_util(image_norm - cv2.erode(image_norm, kernel=np.ones((5,5)),iterations=5))
        quads = {}
        h1, w1 = image_eroded_sub.shape[:2]
        midh,midw = h1//3, w1//2
        origins=[[0,0],[midw,0],[0,midh],[midw,midh]]
        quads[0]=image_eroded_sub[0:midh,0:midw];
        quads[1]=image_eroded_sub[0:midh,midw:w1];
        quads[2]=image_eroded_sub[midh:h1,0:midw];
        quads[3]=image_eroded_sub[midh:h1,midw:w1];

        # Draw Quadlines
        image_eroded_sub[ : , midw:midw+2] = 255
        image_eroded_sub[ midh:midh+2, : ] = 255


        best_scale, allMaxT = getBestMatch(image_eroded_sub)
        if(best_scale == None):
            # TODO: Plot and see performance of markerscaleRange

            return None

        templ = imutils.resize(marker if ERODE_SUB_OFF else marker_eroded_sub, height = int(marker_eroded_sub.shape[0]*best_scale))
        h,w=templ.shape[:2]
        centres = []
        sumT, maxT = 0, 0
        for k in range(0,4):
            res = cv2.matchTemplate(quads[k],templ,cv2.TM_CCOEFF_NORMED)
            maxT = res.max()
            if(maxT < thresholdCircle or abs(allMaxT-maxT) >= thresholdVar):
                # Warning - code will stop in the middle. Keep Threshold low to avoid.
                if(showimglvl>=1):
                    pass
                    #show("no_pts_"+filename,image_eroded_sub,0)
                    #show("res_Q"+str(k+1),res,1)
                return None

            pt=np.argwhere(res==maxT)[0];
            pt = [pt[1],pt[0]]
            pt[0]+=origins[k][0]
            pt[1]+=origins[k][1]
            image_norm = cv2.rectangle(image_norm,tuple(pt),(pt[0]+w,pt[1]+h),(150,150,150),2)
            # display:
            image_eroded_sub = cv2.rectangle(image_eroded_sub,tuple(pt),(pt[0]+w,pt[1]+h),(50,50,50) if ERODE_SUB_OFF else (155,155,155), 4)
            centres.append([pt[0]+w/2,pt[1]+h/2])
            sumT += maxT
        # analysis data
        thresholdCircles.append(sumT/4)

        image_norm = four_point_transform(image_norm, np.array(centres))
        # appendSaveImg(1,image_eroded_sub)
        # appendSaveImg(1,image_norm)

        appendSaveImg(2,image_eroded_sub)
        # res = cv2.matchTemplate(image_eroded_sub,templ,cv2.TM_CCOEFF_NORMED)
        # res[ : , midw:midw+2] = 255
        # res[ midh:midh+2, : ] = 255
        # show("Markers Matching",res)
        if(showimglvl>=2 and showimglvl < 4):
            image_eroded_sub = resize_util_h(image_eroded_sub, image_norm.shape[0])
            image = resize_util_h(image, image_norm.shape[0])
            image_eroded_sub[:,-5:] = 0
            h_stack = np.hstack((image,image_eroded_sub, image_norm))
            #show("Warped: "+filename,resize_util(h_stack,int(display_width*1.6)),0,0,[0,0])
        return image_norm

def drawTemplateLayout(img, template, shifted=True, draw_qvals=False, border=-1):
    img = resize_util(img,template.dims[0],template.dims[1])
    final_align = img.copy()
    boxW,boxH = template.bubbleDims
    for QBlock in template.QBlocks:
        s,d = QBlock.orig, QBlock.dims
        shift = QBlock.shift
        if(shifted):
            cv2.rectangle(final_align,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),CLR_BLACK,3)
        else:
            cv2.rectangle(final_align,(s[0],s[1]),(s[0]+d[0],s[1]+d[1]),CLR_BLACK,3)
        for qStrip, qBoxPts in QBlock.traverse_pts:
            for pt in qBoxPts:
                x,y = (pt.x + QBlock.shift,pt.y) if shifted else (pt.x,pt.y)
                cv2.rectangle(final_align,(int(x+boxW/10),int(y+boxH/10)),(int(x+boxW-boxW/10),int(y+boxH-boxH/10)), CLR_DARK_GRAY,border)
                if(draw_qvals):
                    rect = [y,y+boxH,x,x+boxW]
                    cv2.putText(final_align,'%d'% (cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0]), (rect[2]+2, rect[0] + (boxH*2)//3),cv2.FONT_HERSHEY_SIMPLEX, 0.6,CLR_BLACK,2)
        if(shifted):
            cv2.putText(final_align,'s%s'% (shift), tuple(s - [template.dims[0]//20,-d[1]//2]),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,CLR_BLACK,4)
    return final_align

def getGlobalThreshold(QVals_orig, plotTitle=None, plotShow=True, sortInPlot=True):
    QVals = sorted(QVals_orig)
    l=len(QVals)-1
    max1,thr1=MIN_JUMP,255
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        if(jump > max1):
            max1 = jump
            thr1 = QVals[i-1] + jump/2

    max2,thr2=MIN_JUMP,255
    for i in range(1,l):
        jump = QVals[i+1] - QVals[i-1]
        newThr = QVals[i-1] + jump/2
        if(jump > max2 and abs(thr1-newThr) > JUMP_DELTA):
            max2=jump
            thr2=newThr
    globalTHR, j_low, j_high = thr1, thr1 - max1//2, thr1 + max1//2

    if(plotTitle is not None):
        f, ax = plt.subplots()
        ax.bar(range(len(QVals_orig)),QVals if sortInPlot else QVals_orig);
        ax.set_title(plotTitle)
        thrline=ax.axhline(globalTHR,color='green',ls='--', linewidth=5)
        thrline.set_label("Global Threshold")
        thrline=ax.axhline(thr2,color='red',ls=':', linewidth=3)
        thrline.set_label("THR2 Line")
        ax.set_ylabel("Values")
        ax.set_xlabel("Position")
        ax.legend()
        if(plotShow):
            plt.title(plotTitle)
            plt.show()

    return globalTHR, j_low, j_high

def getLocalThreshold(qNo, QVals, globalTHR, noOutliers, plotTitle=None, plotShow=True):

    QVals= sorted(QVals)

    if(len(QVals) < 3):
        thr1 = globalTHR if np.max(QVals)-np.min(QVals) < MIN_GAP else np.mean(QVals)
    else:
        l=len(QVals)-1
        max1,thr1=MIN_JUMP,255
        for i in range(1,l):
            jump = QVals[i+1] - QVals[i-1]
            if(jump > max1):
                max1 = jump
                thr1 = QVals[i-1] + jump/2
        if(max1 < CONFIDENT_JUMP):
            if(noOutliers):
                thr1 = globalTHR
            else:
                pass


    if(plotShow and plotTitle is not None):
        f, ax = plt.subplots()
        ax.bar(range(len(QVals)),QVals);
        thrline=ax.axhline(thr1,color='green',ls=('-.'), linewidth=3)
        thrline.set_label("Local Threshold")
        thrline=ax.axhline(globalTHR,color='red',ls=':', linewidth=5)
        thrline.set_label("Global Threshold")
        ax.set_title(plotTitle)
        ax.set_ylabel("Bubble Mean Intensity")
        ax.set_xlabel("Bubble Number(sorted)")
        ax.legend()
        if(plotShow):
            plt.show()
    return thr1


def readResponse(image,name,savedir=None,autoAlign=False):
    global clahe
    TEMPLATE = TEMPLATES
    try:
        img = image.copy()
        origDim = img.shape[:2]
        img = resize_util(img,TEMPLATE.dims[0],TEMPLATE.dims[1])
        if(img.max()>img.min()):
            img = normalize_util(img)
        transp_layer = img.copy()
        final_marked = img.copy()

        morph = img.copy()
        appendSaveImg(3,morph)

        if(autoAlign==True):
            morph = clahe.apply(morph)
            appendSaveImg(3,morph)
            morph = adjust_gamma(morph,GAMMA_LOW)
            ret, morph = cv2.threshold(morph,220,220,cv2.THRESH_TRUNC)
            morph = normalize_util(morph)
            appendSaveImg(3,morph)
            if(showimglvl>=4):
                show("morph1",morph,0,1)

        alpha = 0.65
        alpha1 = 0.55

        boxW,boxH = TEMPLATE.bubbleDims
        lang = ['E','H']
        OMRresponse={}

        multimarked,multiroll=0,0

        blackVals=[0]
        whiteVals=[255]

        if(showimglvl>=5):
            allCBoxvals={"Int":[],"Mcq":[]}#"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}
            qNums={"Int":[],"Mcq":[]}#,"QTYPE_ROLL":[]}#,"QTYPE_MED":[]}


        if(autoAlign == True):
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
            morph_v = cv2.morphologyEx(morph, cv2.MORPH_OPEN, v_kernel, iterations=3)
            ret, morph_v = cv2.threshold(morph_v,200,200,cv2.THRESH_TRUNC)
            morph_v = 255 - normalize_util(morph_v)

            if(showimglvl>=3):
                show("morphed_vertical",morph_v,0,1)

            appendSaveImg(3,morph_v)

            morphTHR = 60
            _, morph_v = cv2.threshold(morph_v,morphTHR,255,cv2.THRESH_BINARY)
            morph_v = cv2.erode(morph_v,  np.ones((5,5),np.uint8), iterations = 2)

            appendSaveImg(3,morph_v)
            if(showimglvl>=3):
                show("morph_thr_eroded", morph_v, 0, 1)


            appendSaveImg(6,morph_v)

            for QBlock in TEMPLATE.QBlocks:
                s,d = QBlock.orig, QBlock.dims
                ALIGN_STRIDE, MATCH_COL, ALIGN_STEPS = 1, 5, int(boxW * 2 / 3)
                shift, steps = 0, 0
                THK = 3
                while steps < ALIGN_STEPS:
                    L = np.mean(morph_v[s[1]:s[1]+d[1],s[0]+shift-THK:-THK+s[0]+shift+MATCH_COL])
                    R = np.mean(morph_v[s[1]:s[1]+d[1],s[0]+shift-MATCH_COL+d[0]+THK:THK+s[0]+shift+d[0]])

                    # For demonstration purposes-
                    if(QBlock.key=="Int1"):
                        ret = morph_v.copy()
                        cv2.rectangle(ret,(s[0]+shift-THK,s[1]),(s[0]+shift+THK+d[0],s[1]+d[1]),CLR_WHITE,3)
                        appendSaveImg(6,ret)
                    LW,RW= L > 100, R > 100
                    if(LW):
                        if(RW):
                            break
                        else:
                            shift -= ALIGN_STRIDE
                    else:
                        if(RW):
                            shift += ALIGN_STRIDE
                        else:
                            break
                    steps += 1

                QBlock.shift = shift

        final_align = None
        if(showimglvl>=2):
            initial_align = drawTemplateLayout(img, TEMPLATE, shifted=False)
            final_align = drawTemplateLayout(img, TEMPLATE, shifted=True, draw_qvals=True)
            # appendSaveImg(4,mean_vals)
            appendSaveImg(2,initial_align)
            appendSaveImg(2,final_align)
            appendSaveImg(5,img)
            if(autoAlign == True):
                final_align = np.hstack((initial_align, final_align))

        # Get mean vals n other stats
        allQVals, allQStripArrs, allQStdVals =[], [], []
        totalQStripNo = 0
        for QBlock in TEMPLATE.QBlocks:
            QStdVals=[]
            for qStrip, qBoxPts in QBlock.traverse_pts:
                QStripvals = []
                for pt in qBoxPts:
                    # shifted
                    x,y = (pt.x + QBlock.shift,pt.y)
                    rect = [y,y+boxH,x,x+boxW]
                    QStripvals.append(cv2.mean(img[  rect[0]:rect[1] , rect[2]:rect[3] ])[0])
                QStdVals.append(round(np.std(QStripvals),2))
                allQStripArrs.append(QStripvals)
                allQVals.extend(QStripvals)
                totalQStripNo+=1
            allQStdVals.extend(QStdVals)
        globalStdTHR, jstd_low, jstd_high = getGlobalThreshold(allQStdVals)#, "Q-wise Std-dev Plot", plotShow=True, sortInPlot=True)
        globalTHR, j_low, j_high = getGlobalThreshold(allQVals)#, "Mean Intensity Histogram", plotShow=True, sortInPlot=True)


        perOMRThresholdAvg, totalQStripNo, totalQBoxNo = 0, 0, 0
        for QBlock in TEMPLATE.QBlocks:
            blockQStripNo = 1 # start from 1 is fine here
            shift=QBlock.shift
            s,d = QBlock.orig, QBlock.dims
            key = QBlock.key[:3]
            # cv2.rectangle(final_marked,(s[0]+shift,s[1]),(s[0]+shift+d[0],s[1]+d[1]),CLR_BLACK,3)
            for qStrip, qBoxPts in QBlock.traverse_pts:
                # All Black or All White case
                noOutliers = allQStdVals[totalQStripNo] < globalStdTHR
                perQStripThreshold = getLocalThreshold(qBoxPts[0].qNo, allQStripArrs[totalQStripNo],
                    globalTHR, noOutliers,
                    "Mean Intensity Histogram for "+ key +"."+ qBoxPts[0].qNo+'.'+str(blockQStripNo),
                    showimglvl>=6)
                perOMRThresholdAvg += perQStripThreshold

                for pt in qBoxPts:
                    x,y = (pt.x + QBlock.shift,pt.y)
                    boxval0 = allQVals[totalQBoxNo]
                    detected = perQStripThreshold > boxval0
                    if (detected):
                        cv2.rectangle(final_marked,(int(x+boxW/12),int(y+boxH/12)),(int(x+boxW-boxW/12),int(y+boxH-boxH/12)), CLR_DARK_GRAY, 3)
                    else:
                        cv2.rectangle(final_marked,(int(x+boxW/10),int(y+boxH/10)),(int(x+boxW-boxW/10),int(y+boxH-boxH/10)), CLR_GRAY,-1)


                    # TODO Make this part useful! (Abstract visualizer to check status)
                    if (detected):
                        q, val = pt.qNo, str(pt.val)
                        cv2.putText(final_marked,val,(x,y),cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE,(20,20,10),int(1+3.5*TEXT_SIZE))
                        # Only send rolls multi-marked in the directory
                        multimarkedL = q in OMRresponse
                        multimarked = multimarkedL or multimarked
                        OMRresponse[q] = (OMRresponse[q] + val) if multimarkedL else val
                        multiroll = multimarkedL and 'roll' in str(q)
                        blackVals.append(boxval0)
                    else:
                        whiteVals.append(boxval0)

                    totalQBoxNo+=1
                    # /for qBoxPts
                # /for qStrip

                if( showimglvl>=5):
                    if(key in allCBoxvals):
                        qNums[key].append(key[:2]+'_c'+str(blockQStripNo))
                        allCBoxvals[key].append(allQStripArrs[totalQStripNo])

                blockQStripNo += 1
                totalQStripNo += 1
            # /for QBlock
        if(totalQStripNo==0):
            exit(7)
        perOMRThresholdAvg /= totalQStripNo
        perOMRThresholdAvg = round(perOMRThresholdAvg,2)
        # Translucent
        cv2.addWeighted(final_marked,alpha,transp_layer,1-alpha,0,final_marked)
        # Box types
        if( showimglvl>=5):
            # plt.draw()
            f, axes = plt.subplots(len(allCBoxvals),sharey=True)
            f.canvas.set_window_title(name)
            ctr=0
            typeName={"Int":"Integer","Mcq":"MCQ","Med":"MED","Rol":"Roll"}
            for k,boxvals in allCBoxvals.items():
                axes[ctr].title.set_text(typeName[k]+" Type")
                axes[ctr].boxplot(boxvals)
                # thrline=axes[ctr].axhline(perOMRThresholdAvg,color='red',ls='--')
                # thrline.set_label("Average THR")
                axes[ctr].set_ylabel("Intensity")
                axes[ctr].set_xticklabels(qNums[k])
                # axes[ctr].legend()
                ctr+=1
            # imshow will do the waiting
            plt.tight_layout(pad=0.5)
            plt.show()

        if(showimglvl>=3 and final_align is not None):
            final_align = resize_util_h(final_align,int(display_height))
            show("Template Alignment Adjustment", final_align, 0, 0)# [final_align.shape[1],0])

        # TODO: refactor "type(savedir) != type(None) "
        if (saveMarked and type(savedir) != type(None) ):
            if(multiroll):
                savedir = savedir+'_MULTI_/'
            saveImg(savedir+name, final_marked)

        if(showimglvl>=1):
            # final_align = resize_util_h(final_align,int(display_height))
            # show("Final Alignment : "+name,final_align,0,0)
            show("Final Marked Bubbles : "+name,resize_util_h(final_marked,int(display_height*1.3)),1,1)

        appendSaveImg(2,final_marked)

        # saveImgList[3] = [hist, final_marked]

        for i in range(saveimglvl):
            saveOrShowStacks(i+1, name, savedir)

        return OMRresponse,final_marked,multimarked,multiroll

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]


#end imports from utils.py


#imports from main.py
def move(error_code, filepath,filepath2):
    global filesNotMoved
    filesNotMoved += 1
    return True
    # if(error_code!=NO_MARKER_ERR):

    global filesMoved
    if(not os.path.exists(filepath)):
        return False
    if(os.path.exists(filepath2)):
        return False

    os.rename(filepath,filepath2)
    filesMoved+=1
    return True


def processOMR( omrResp):
    resp={}
    UNMARKED = '' # 'X'

    for qNo, respKeys in TEMPLATES.concats.items():
        resp[qNo] = ''.join([omrResp.get(k,UNMARKED) for k in respKeys])

    for qNo in TEMPLATES.singles:
        resp[qNo] = omrResp.get(qNo,UNMARKED)
    return resp

def report(Status,streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks):
    pass

def evaluate(resp, answers, explain=False):
    global Sections

    marks = 0
    #answers = Answers
    if(explain):
        pass
    for scheme,section in Sections.items():
        sectionques = len(answers)
        prevcorrect=None
        allflag=1
        streak=0
        for q in range(len(answers)):
            qnum = q+1
            qNo = 'q'+ str(qnum)
            ans = answers[qNo]
            marked = resp.get(qNo, 'X')
            unmarked = marked=='X' or marked==''
            bonus = 'BONUS' in ans
            correct = bonus or (marked in ans[0])
            inrange=0

            """if(unmarked or int(q)==firstQ):
                streak=0
            elif(prevcorrect == correct):
                streak+=1
            else:
                streak=0"""

            if True:
                currmarks = ans[1] if correct else ans[2]

            elif( 'allNone' in scheme):
                allflag = allflag and correct
                if(q == lastQ ):
                    prevcorrect = correct
                    currmarks = section['marks'] if allflag else 0
                else:
                    currmarks = 0

            elif('Proxy' in scheme):
                a=int(ans[0])
                #proximity check
                inrange = 1 if unmarked else (float(abs(int(marked) - a))/float(a) <= 0.25)
                currmarks = section['+marks'] if correct else (0 if inrange else -section['-marks'])

            elif('Fibo' in scheme or 'Power' in scheme or 'Boom' in scheme):
                currmarks = section['+seq'][streak] if correct else (0 if unmarked else -section['-seq'][streak])
            elif('TechnoFin' in scheme):
                currmarks = 0
            else:
                pass
            prevmarks=marks
            marks += int(currmarks)

            if(explain):
                if bonus:
                    report('BonusQ',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif correct:
                    report('Correct',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif unmarked:
                    report('Unmarked',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif inrange:
                    report('InProximity',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                else:
                    report('Incorrect',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)

            prevcorrect = correct

    return marks


#end imports from main.py

# if you do not want pretty printing, just use pandas:
# df.to_html(intermediate_html)

#excel to pdf
def xl2pdf(df, pdf,  title = 'Result'):
    HTML_TEMPLATE1 = '''
    <html>
    <head>
    <style>
      h2 {
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
      }
      table {
        margin-left: auto;
        margin-right: auto;
      }
      table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
      }
      th, td {
        padding: 5px;
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
        font-size: 90%;
      }
      table tbody tr:hover {
        background-color: #dddddd;
      }
      .wide {
        width: 90%;
      }
    </style>
    </head>
    <body>
    '''

    HTML_TEMPLATE2 = '''
    </body>
    </html>
    '''

    ht = ''
    if title != '':
        ht += '<h2> %s </h2>\n' % title

    #heighest_marks = df['Total Marks'].max()
    #avg_marks = str(df['Total Marks'].mean())[:5]

    ht += '<h4>Highest Marks: %s </h4>' % df['Total Marks'].max()
    ht += '<h4>Average Marks: %s </h4>' % str(df['Total Marks'].mean())[:5]
    ht += '<h4>Total Students: %s </h4>' % len(df)

    ht += df.to_html(classes='wide', escape=False)
    print(os.getcwd())

    try:
        os.remove(r"./functions/media/output/htmlpdftemp/temp.txt")
        os.remove(r"./functions/media/output/htmlpdftemp/temp.html")
    except:
        pass

    rlist = ht[:].split('\n')
    with open(r"./functions/media/output/htmlpdftemp/temp.txt", 'w') as f:
         f.write(ht)

    with open(r"./functions/media/output/htmlpdftemp/temp.html", 'w') as f:
         f.write(HTML_TEMPLATE1 + ht + HTML_TEMPLATE2)

    #path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    #config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

    pdfkit.from_file(r"./functions/media/output/htmlpdftemp/temp.html", pdf)#, configuration=config)

    return rlist


#reading answer_key
def fromImage(imgPath):
    inOMR = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    OMRcrop = getROI(inOMR, noCropping=args["noCropping"], noMarkers=args["noMarkers"])
    if(OMRcrop is None):
        return "wrong file"
    OMRresponseDict, final_marked, MultiMarked, multiroll = readResponse(OMRcrop,name = imgPath.split('/')[-1], autoAlign=args["autoAlign"])
    #mainDict = {}
    #for k, v in OMRresponseDict.items():
    #    mainDict[k] = [v, str(answers['data'][int(k[1:])-1][1]),str(answers['data'][int(k[1:])-1][2]), str(answers['data'][int(k[1:])-1][3])]
    return OMRresponseDict#mainDict

#reading answer_key
def extractAnswers(csvPath = None, imgPath = None):
    if imgPath == "default.jpg":
        answerdf = pd.read_csv(csvPath)
        for i in range(len(answerdf)):
            answerdf['Question_no'].iloc[i] = int(i + 1)
        answerdf['Question_no'] = answerdf['Question_no'].astype(int)
        answerdf = answerdf.set_index('Question_no')
        answers = answerdf.to_dict('split')
        mainDict = {}
        index = 0
        for i in range(len(answers['index'])):
            mainDict['q{}'.format(i+1)] = answers['data'][i]
            index += 1

        return mainDict
    else:
        answerdf = pd.read_csv(csvPath)
        for i in range(len(answerdf)):
            answerdf['Question_no'].iloc[i] = int(i + 1)
        answerdf['Question_no'] = answerdf['Question_no'].astype(int)
        answerdf = answerdf.set_index('Question_no')
        answers = answerdf.to_dict('split')

        imgPath = r"media/" + str(imgPath)
        inOMR = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        OMRcrop = getROI(inOMR, noCropping=args["noCropping"], noMarkers=args["noMarkers"])
        if(OMRcrop is None):
            return "wrong file"
        OMRresponseDict, final_marked, MultiMarked, multiroll = readResponse(OMRcrop,name = imgPath.split('/')[-1], autoAlign=args["autoAlign"])
        mainDict = {}
        for k, v in OMRresponseDict.items():
            mainDict[k] = [v, str(answers['data'][int(k[1:])-1][1]),str(answers['data'][int(k[1:])-1][2]), str(answers['data'][int(k[1:])-1][3])]
        return mainDict


def getRanks(df):
    df.sort_values('Total Marks', axis=0, ascending=True, kind='quicksort', na_position='last')
    df['Rank'] = df['Total Marks'].rank(ascending = False, method = 'min').astype(int)
    return df



#main function called by external application
args = {"noCropping":True, "noMarkers":True, "autoAlign":False, "setLayout":False}

def main(answers, imagePathListOrDirectory, directory=False):
    global args, filesNotMoved, filesMoved
    #Answers = answers
    pa = r"/home/danniel/b691009ff/src/functions/inputs/OMR_Files/MobileCameraBased/JE"
    #make a list of paths of all the images to be graded
    if directory:
        allOMRs = list(glob.iglob(imagePathListOrDirectory+'/*.jpg')) + list(glob.iglob(imagePathListOrDirectory+'/*.png'))
    else:
        allOMRs = imagePathListOrDirectory



    timeNowHrs=strftime("%I%p",localtime())
    OUTPUT_SET, respCols, emptyResp, filesObj, filesMap = [], {}, {}, {}, {}

    KEY_FN_SORTING = lambda x: int(x[1:]) if ord(x[1]) in range(48,58) else 0
    respCols = sorted( list(TEMPLATES.concats.keys()) + TEMPLATES.singles, key=KEY_FN_SORTING)
    emptyResp = ['']*len(respCols)
    sheetCols = ['Roll Number','Total Marks']
    filesObj = {}


    dis_string = ''.join(random.choices('0123456789abcdefghijklmnopqrstuvwxyz', k = 10))

    filesMap = {
        "Results_csv": resultDir+'Results_'+dis_string+'.csv',
        "MultiMarked_csv": manualDir+'MultiMarkedFiles_'+dis_string+'.csv',
        "Errors_csv": manualDir+'ErrorFiles_'+dis_string+'.csv',
        "BadRollNos_csv": manualDir+'BadRollNoFiles_'+dis_string+'.csv',

        "Results_xlsx": resultDir+'Results_'+dis_string+'.xlsx',
        "MultiMarked_xlsx": manualDir+'MultiMarkedFiles_'+dis_string+'.xlsx',
        "Errors_xlsx": manualDir+'ErrorFiles_'+dis_string+'.xlsx',
        "BadRollNos_xlsx": manualDir+'BadRollNoFiles_'+dis_string+'.xlsx',

        "Results_pdf": resultDir+'Results_'+dis_string+'.pdf',
        "MultiMarked_pdf": manualDir+'MultiMarkedFiles_'+dis_string+'.pdf',
        "Errors_pdf": manualDir+'ErrorFiles_'+dis_string+'.pdf',
        "BadRollNos_pdf": manualDir+'BadRollNoFiles_'+dis_string+'.pdf',
    }

    masterDf = pd.DataFrame(columns = sheetCols, dtype = "object")

    squadlang="XXdummySquad"
    inputFolderName="dummyFolder"

    filesCounter=0
    mws, mbs = [],[]

    if(PRELIM_CHECKS):
        # TODO: add more using unit testing
        TEMPLATE = TEMPLATES
        ALL_WHITE = 255 * np.ones((TEMPLATE.dims[1],TEMPLATE.dims[0]), dtype='uint8')
        OMRresponseDict,final_marked,MultiMarked,multiroll = readResponse("H",ALL_WHITE,name = "ALL_WHITE", savedir = None, autoAlign=False)
        if(OMRresponseDict!={}):
            exit(2)
        ALL_BLACK = np.zeros((TEMPLATE.dims[1],TEMPLATE.dims[0]), dtype='uint8')
        OMRresponseDict,final_marked,MultiMarked,multiroll = readResponse("H",ALL_BLACK,name = "ALL_BLACK", savedir = None, autoAlign=False)
        #show("Confirm : All bubbles are black",final_marked,1,1)



    for filepath in allOMRs:
        if 'gitkeep' in filepath:
            continue
        filesCounter+=1
        filepath = filepath.replace(os.sep,'/')

        filename = re.search(r'.*/(.*)',filepath,re.IGNORECASE).groups()[0]
        inOMR = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        OMRcrop = getROI(inOMR,filename, noCropping=args["noCropping"], noMarkers=args["noMarkers"])
        if(OMRcrop is None):
            newfilepath = errorsDir+filename
            OUTPUT_SET.append([filename]+emptyResp)
            if(move(NO_MARKER_ERR, filepath, newfilepath)):
                err_line = [filename,"NA"]+emptyResp
                pd.DataFrame(err_line, dtype=str).T.to_csv(filesObj["Errors_csv"], quoting = QUOTE_NONNUMERIC,header=False,index=False)
                pd.DataFrame(err_line, dtype=str).T.to_excel(filesObj["Errors_xlsx"], quoting = QUOTE_NONNUMERIC,header=False,index=False)
            continue

        newfilename = inputFolderName + '_' + filename
        savedir = saveMarkedDir + squadlang
        OMRresponseDict,final_marked,MultiMarked,multiroll = readResponse(OMRcrop,name = newfilename, savedir = savedir, autoAlign=args["autoAlign"])

        resp = processOMR(OMRresponseDict)
        score = evaluate(resp, answers, explain=explain)
        respArray=[]
        for k in respCols:
            respArray.append(resp[k])

        OUTPUT_SET.append([filename]+respArray)
        os.remove(filepath)


        if(MultiMarked == 0):
            filesNotMoved+=1;
            newfilepath = savedir+newfilename
            results_line = [filename,score]
            masterDf = masterDf.append(pd.DataFrame([results_line], columns = sheetCols), ignore_index = True)
        else:
            results_line = [filename,0]+['invalid image' for i in range(len(respArray))]

    masterDf = getRanks(masterDf)
    masterDf.to_csv(filesMap["Results_csv"], index = False)
    masterDf.to_excel(filesMap["Results_xlsx"], index = False)
    table = xl2pdf(masterDf, filesMap["Results_pdf"])
    return filesMap["Results_xlsx"], filesMap["Results_pdf"], filesMap["Results_csv"], table
