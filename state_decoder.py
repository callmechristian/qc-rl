import numpy as np

def extract_state(I):
    I, remaining_lives = preprocess(I)
    found_ball = False
    ball_pos = [0,0]
    for i in range(0,len(I) - 4): # the last 3 rows have the block
        # 13,14,15 row is a red tile row we skip this row as its the same color as the ball
        for j in range(0,len(I[i])):
            if I[i][j] == 114 and i == 14 :
                continue
            if I[i][j] == 114 :
                if I[i+1][j] == 114:
                    found_ball = True
                    ball_pos = [i,j]
                    if I[i+2][j] == 114:
                        found_ball = False
                        ball_pos = [0,0]
                        if I[i+3][j] == 114 and I[i+4][j] == 114:
                            found_ball = True
                            ball_pos = [i,j]
    player_pos = 0
    for i in range(len(I)-3,len(I)): # find the position of the player in the last 3 rows
        for j in range(0,len(I[i])-2): # 8 is the length of the player
            # print(I[i][j:j+8],end=" ")
            if(np.sum(I[i][j:j+2]) == 114*2):
                player_pos = j+2
                break
        # print()
    # print(ball_pos,player_pos)
    return [(ball_pos[0]*72+ball_pos[1])/(82*72),(player_pos)/72, remaining_lives]

def preprocess(img):
    img = to_grayscale(downsample(img))
    remaining_lives = read_lives(img)
    img = img[9+7:(len(img)-7)] # 9 is the score part which we crop and 7 border size in top and bottom
    for i in range(4): # border size is 4
        img = np.delete(img,0,1)
        img = np.delete(img,len(img[0])-1,1)
    return img, remaining_lives

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def read_lives(img):
    img = img[0:9] # 9 is the score part

    sample = [img[n][50] for n in range(0,8)]
    if sample == [0, 0, 0, 142, 142, 142, 0, 142]: # 5
        remaining_lives = 5
    elif sample == [0, 0, 0, 142, 142, 142, 0, 0]: # 4
        remaining_lives = 4
    elif sample == [0, 0, 0, 142, 0, 0, 0, 142]: # 3 
        remaining_lives = 3
    elif sample == [0, 0, 0, 142, 0, 142, 142, 142]: # 2 
        remaining_lives = 2
    elif sample == [0, 0, 0, 0, 0, 0, 0, 0]: # 1
        remaining_lives = 1

    return remaining_lives