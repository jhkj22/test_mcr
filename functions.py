def Zhang_Suen_thinning(binary_image):
    image_thinned = binary_image.copy()
    changing_1 = changing_2 = [1]
    while changing_1 or changing_2:
        changing_1 = []
        rows, columns = image_thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns -1):
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and
                    2 <= sum(neighbour_points) <= 6 and
                    count_transition(neighbour_points) == 1 and
                    p2 * p4 * p6 == 0 and
                    p4 * p6 * p8 == 0):
                    changing_1.append((x,y))
        for x, y in changing_1:
            image_thinned[x][y] = 0
        changing_2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns -1):
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and
                    2 <= sum(neighbour_points) <= 6 and
                    count_transition(neighbour_points) == 1 and
                    p2 * p4 * p8 == 0 and
                    p2 * p6 * p8 == 0):
                    changing_2.append((x,y))
        for x, y in changing_2:
            image_thinned[x][y] = 0        

    return image_thinned

def black_one(binary):
    bool_image = binary.astype(bool)
    inv_bool_image = ~bool_image
    return inv_bool_image.astype(int)

def padding_zeros(image):
    import numpy as np
    m,n = np.shape(image)
    padded_image = np.zeros((m+2,n+2))
    padded_image[1:-1,1:-1] = image
    return padded_image

def neighbours(x, y, image):
    return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1],
             image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]]

def count_transition(neighbours):
    neighbours += neighbours[:1]
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(neighbours, neighbours[1:]) )

def inv_black_one(inv_bool_image):
    bool_image = ~inv_bool_image.astype(bool)
    return bool_image.astype(int) * 255

