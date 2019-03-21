import os

def points2pcd(points):
    PCD_DIR_PATH=os.path.join(os.path.abspath('.'),'pcd')
    PCD_FILE_PATH=os.path.join(PCD_DIR_PATH,'cache.pcd')
    if os.path.exists(PCD_FILE_PATH):
    	os.remove(PCD_FILE_PATH)
    handle = open(PCD_FILE_PATH, 'a')

    point_num=points.shape[0]

    # headers
    handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # data
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)
    handle.close()
    os.system('pcl_viewer %s' %PCD_FILE_PATH)

