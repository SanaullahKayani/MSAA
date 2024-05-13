import numpy as np

class Fragment:
    def __init__(self, x : int, y : int, depth : float, interpolated_data, ratio : float ):
        self.x = x
        self.y = y
        self.depth = depth
        self.ratio = ratio
        self.interpolated_data = interpolated_data
        self.output = []

def edgeSide(p, v0, v1) : 
    return (p[0]-v0[0])*(v1[1]-v0[1]) - (p[1]-v0[1])*(v1[0]-v0[0])

def edgeSide3D(p,v0,v1) :
    return np.linalg.norm(np.cross(p[0:3]-v0[0:3],v1[0:3]-v0[0:3]))

class GraphicPipeline:
    def __init__ (self, width, height, num_samples = 16):
        self.width = width
        self.height = height
        self.num_samples = num_samples
        self.image = np.zeros((height, width, 3))
        self.depthBuffer = np.ones((height, width))


    def VertexShader(self, vertices, data) :
        outputVertices = np.zeros((vertices.shape[0],12))
        for i in range(vertices.shape[0]) :
            x = vertices[i][0]
            y = vertices[i][1]
            z = vertices[i][2]
            w = 1.0

            vec = np.array([[x],[y],[z],[w]])

            vec = np.matmul(data['projMatrix'],np.matmul(data['viewMatrix'],vec))

            outputVertices[i][0] = vec[0]/vec[3]
            outputVertices[i][1] = vec[1]/vec[3]
            outputVertices[i][2] = vec[2]/vec[3]

            outputVertices[i][3] = vertices[i][3]
            outputVertices[i][4] = vertices[i][4]
            outputVertices[i][5] = vertices[i][5]

            outputVertices[i][6] = data['cameraPosition'][0] - vertices[i][0]
            outputVertices[i][7] = data['cameraPosition'][1] - vertices[i][1]
            outputVertices[i][8] = data['cameraPosition'][2] - vertices[i][2]

            outputVertices[i][9] = data['lightPosition'][0] - vertices[i][0]
            outputVertices[i][10] = data['lightPosition'][1] - vertices[i][1]
            outputVertices[i][11] = data['lightPosition'][2] - vertices[i][2]

        return outputVertices


    def Rasterizer(self, v0, v1, v2) :
        fragments = []

        #culling back face
        area = edgeSide(v0,v1,v2)
        area3D = edgeSide3D(v0,v1,v2)
        if area < 0 :
            return fragments
        
        
        #AABBox computation
        #compute vertex coordinates in screen space
        v0_image = np.array([0,0])
        v0_image[0] = (v0[0]+1.0)/2.0 * self.width 
        v0_image[1] = ((v0[1]+1.0)/2.0) * self.height 

        v1_image = np.array([0,0])
        v1_image[0] = (v1[0]+1.0)/2.0 * self.width 
        v1_image[1] = ((v1[1]+1.0)/2.0) * self.height 

        v2_image = np.array([0,0])
        v2_image[0] = (v2[0]+1.0)/2.0 * self.width 
        v2_image[1] = (v2[1]+1.0)/2.0 * self.height 

        #compute the two point forming the AABBox
        A = np.min(np.array([v0_image,v1_image,v2_image]), axis = 0)
        B = np.max(np.array([v0_image,v1_image,v2_image]), axis = 0)

        #cliping the bounding box with the borders of the image
        max_image = np.array([self.width-1,self.height-1])
        min_image = np.array([0.0,0.0])

        A  = np.max(np.array([A,min_image]),axis = 0)
        B  = np.min(np.array([B,max_image]),axis = 0)
        
        #cast bounding box to int
        A = A.astype(int)
        B = B.astype(int)
        #Compensate rounding of int cast
        B = B + 1

        #for each pixel in the bounding box
        for j in range(A[1], B[1]) : 
           for i in range(A[0], B[0]) :
                num_inside = 0
                x = (i+0.5)/self.width * 2.0 - 1 
                y = (j+0.5)/self.height * 2.0 - 1

                p = np.array([x,y])
                
                area0 = edgeSide(p,v0,v1)
                area1 = edgeSide(p,v1,v2)
                area2 = edgeSide(p,v2,v0)

                #test if p is inside the triangle
                if (area0 >= 0 and area1 >= 0 and area2 >= 0) : 
                    
                    #Computing 2d barricentric coordinates
                    lambda0 = area1/area
                    lambda1 = area2/area
                    lambda2 = area0/area
                    
                    one_over_z = lambda0 * 1/v0[2] + lambda1 * 1/v1[2] + lambda2 * 1/v2[2]
                    z = 1/one_over_z
                    
                    p = np.array([x,y,z])
                    
                    #Recomputing the barricentric coordinaties for vertex interpolation
                    area0 = edgeSide3D(p,v0,v1)
                    area1 = edgeSide3D(p,v1,v2)
                    area2 = edgeSide3D(p,v2,v0)

                    lambda0 = area1/area3D
                    lambda1 = area2/area3D
                    lambda2 = area0/area3D
                    
                    l = v0.shape[0]
                    #interpolating
                    interpolated_data = v0[3:l] * lambda0 + v1[3:l] * lambda1 + v2[3:l] * lambda2
                    
                    # Loop over all samples within the pixel
                    for s in range(self.num_samples):
                        sample_offset_x = (s % 2) * 0.5
                        sample_offset_y = (s // 2) * 0.5
                        sample_x = (i + 0.25 + sample_offset_x) / self.width * 2.0 - 1.0  # Adjusted sampling within the pixel
                        sample_y = (j + 0.25 + sample_offset_y) / self.height * 2.0 - 1.0
        
                        p = np.array([sample_x, sample_y])
        
                        area0 = edgeSide(p, v0, v1)
                        area1 = edgeSide(p, v1, v2)
                        area2 = edgeSide(p, v2, v0)
        
                        # If the sample point is inside the triangle, increment the counter
                        if area0 >= 0 and area1 >= 0 and area2 >= 0:
                            num_inside += 1
                            
                    # Calculate the ratio of covered samples to total samples
                    ratio = num_inside / self.num_samples
                    #Emiting Fragment
                    fragments.append(Fragment(i,j,z, interpolated_data, ratio))

        return fragments
    
    def fragmentShader(self,fragment,data):
        #unpacking and normalizing interpolated data
        N = fragment.interpolated_data[0:3]
        N = N/np.linalg.norm(N)
        V = fragment.interpolated_data[3:6]
        V = V/np.linalg.norm(V)
        L = fragment.interpolated_data[6:9]
        L = L/np.linalg.norm(L)

        # reflected ray
        R = 2 * np.dot(L,N) * N  -L
        
        # computing the different component of the phong illumination
        ambient = 1.0
        diffuse = max(np.dot(N,L),0)
        specular = np.power(max(np.dot(R,V),0.0),64) 

        ka = 0.1
        kd = 0.9
        ks = 0.3

        #mixing the component
        phong = ka * ambient + kd * diffuse + ks * specular

        #applying the toon effect
        phong = np.ceil(phong*4 +1 )/6.0

        color = np.array([phong,phong,phong])

        fragment.output = color * fragment.ratio

    def draw(self, vertices, triangles, data):
        #Calling vertex shader
        newVertices = self.VertexShader(vertices, data)
        
        fragments = []
        #Calling Rasterizer
        for i in triangles :
            fragments.extend(self.Rasterizer(newVertices[i[0]], newVertices[i[1]], newVertices[i[2]]))
        
        for f in fragments:
            self.fragmentShader(f,data)
            #depth test
            if self.depthBuffer[f.y][f.x] > f.depth : 
                self.depthBuffer[f.y][f.x] = f.depth
                
                self.image[f.y][f.x] = f.output
