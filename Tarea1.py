
#Tarea 1

#Renzo Zanca R.

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
import grafica.scene_graph as sg

#funciones para crear esferas y anillos:

def createColorSphere(N, R, r, g, b):
    
    #inicializar:
    vertices = []
    indices = []

    dtheta = 2 * np.pi / N 
    dv = np.pi / N

    for i in range(N):
        for j in range(N):
            theta = i * dtheta 
            v = j * dv - np.pi/2

            #vertices coordenadas esfericas
            vertices += [R * np.cos(theta) * np.cos(v), R * np.sin(theta) * np.cos(v), R * np.sin(v), r, g, b]

            # se forman un cuadrilatero (2 triangulos) para cada iteracion:
            indices += [i*N+j, (i*N)+j+1, (i+1)*N+j]
            indices += [(i*N)+j+1, (i+1)*N+j, (i+1)*N+j+1] 

        # Agregamos los ultimos indices
        indices += [N*N, N*N + 1, 0]
        indices += [N*N + 1, 0, 1]


    return bs.Shape(vertices, indices)

def createColorRing(N,R1,R2,r,g,b):
    #se crearan 2 circulos concentricos:

    #inicializar:
    vertices = []
    indices = []

    dc1 = 2 * np.pi / N 
    dc2 = 2 * np.pi / N 

    for i in range(0,N,2):
        c1 = i * dc1
        c2 = i * dc2

        vertices +=  [R1 * np.cos(c1), R1 * np.sin(c1), 0, r, g, b]
        vertices +=  [R2 * np.cos(c2), R2 * np.sin(c2), 0, r, g, b]

        # se forman un cuadrilatero (2 triangulos) para cada iteracion:
        indices += [i, i+1, i+3]
        indices += [i, i+2, i+3] 

    # Agregamos los ultimos indices
    indices += [N, N + 1, 0]
    indices += [N + 1, 0, 1] #[N, N+1, 1, N, 0, 1]
 
    return bs.Shape(vertices,indices)

#funcion para crear formas en el GPU:
def createGPUShape(shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpuShape

#para crear el sol la tierra y la luna usare un grafo de escena. Reutilice el codigo de ex_scene_graph_solar

def createSystem(pipeline):

    sunShape = createGPUShape(createColorSphere(50,1,250/255,190/255,30/255))
    earthShape = createGPUShape(createColorSphere(50,1,0,90/255,142/255))
    moonShape = createGPUShape(createColorSphere(50,1,240/255,240/255,240/255))

    sunNode = sg.SceneGraphNode("sunNode")
    sunNode.transform = tr.uniformScale(1.25)
    sunNode.childs += [sunShape]

    earthNode = sg.SceneGraphNode("earthNode")
    earthNode.transform = tr.uniformScale(0.2)
    earthNode.childs += [earthShape]

    moonNode = sg.SceneGraphNode("moonNode")
    moonNode.transform = tr.uniformScale(0.03)
    moonNode.childs += [moonShape]

    moonRotation = sg.SceneGraphNode("moonRotation")
    moonRotation.childs += [moonNode]

    earthRotation = sg.SceneGraphNode("earthRotation")
    earthRotation.childs += [earthNode]

    moonRotation = sg.SceneGraphNode("moonRotation")
    moonRotation.childs += [moonNode]

    sunRotation = sg.SceneGraphNode("sunRotation")
    sunRotation.childs += [sunNode]
    
    moonPosition = sg.SceneGraphNode("moonSystem")
    moonPosition.transform = tr.translate(0.5,0.0,0.0)
    moonPosition.childs += [moonRotation] 

    moonSystem = sg.SceneGraphNode("moonSystem")
    moonSystem.childs += [moonPosition]
    
    earthPosition = sg.SceneGraphNode("earthSystem")
    earthPosition.transform = tr.translate(3.7, 0.0, 0.0)
    earthPosition.childs += [earthRotation]
    earthPosition.childs += [moonSystem]

    earthSystem = sg.SceneGraphNode("earthSystem")
    earthSystem.childs += [earthPosition]

    systemNode = sg.SceneGraphNode("solarSystem")
    systemNode.childs += [sunRotation]
    systemNode.childs += [earthSystem]
    
    return systemNode

# Initialize glfw
if not glfw.init():
    glfw.set_window_should_close(window, True)

#crear ventana
width = 700
height = 700
title = "Sistema solar"
window = glfw.create_window(width, height, title, None, None)

if not window:
    glfw.terminate()
    glfw.set_window_should_close(window, True)

glfw.make_context_current(window)

#creamos 2 pipeline, uno para la textura del fondo y otro para el resto:
texturePipeline = es.SimpleTextureModelViewProjectionShaderProgram()
pipeline = es.SimpleModelViewProjectionShaderProgram()

glUseProgram(texturePipeline.shaderProgram)
glUseProgram(pipeline.shaderProgram)

# Setting up the clear screen color
glClearColor(0.15, 0.15, 0.15, 1.0)

# Enabling transparencies
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

#fondo de estrellas
#guardar imagen en la carpeta assets
backgroundShape = bs.createTextureQuad(1,1)
bs.scaleVertices(backgroundShape, 5, [2,2,1])
gpuBackground = es.GPUShape().initBuffers()
texturePipeline.setupVAO(gpuBackground)
gpuBackground.fillBuffers(backgroundShape.vertices, backgroundShape.indices, GL_STATIC_DRAW)
gpuBackground.texture = es.textureSimpleSetup(
    getAssetPath("estrellas.jpg"), GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR)

#creamos las figuras en el GPU:

gpuMercurio = createGPUShape(createColorSphere(50,1,161/255,130/255,98/255))
gpuVenus = createGPUShape(createColorSphere(50,1,243/255,135/255,7/255))
gpuMarte = createGPUShape(createColorSphere(50,1,190/255,36/255,0))
gpuJupiter = createGPUShape(createColorSphere(50,1,242/255,217/255,176/255))
gpuSaturno = createGPUShape(createColorSphere(50,1,204/255,157/255,124/255))
gpuUrano = createGPUShape(createColorSphere(50,1,51/255,170/255,206/255))
gpuNeptuno = createGPUShape(createColorSphere(50,1,57/255,60/255,100/255))

#anillos saturno:
gpuAnillo1 = createGPUShape(createColorRing(500,1.5,1.52,253/255,253/255,253/255)) 
gpuAnillo2 = createGPUShape(createColorRing(500,1.53,1.55,243/255,243/255,243/255)) 
gpuAnillo3 = createGPUShape(createColorRing(500,1.56,1.58,233/255,233/255,233/255)) 
gpuAnillo4 = createGPUShape(createColorRing(500,1.59,1.61,223/255,223/255,223/255)) 
gpuAnillo5 = createGPUShape(createColorRing(500,1.62,1.64,213/255,213/255,213/255)) 
gpuAnillo6 = createGPUShape(createColorRing(500,1.65,1.67,203/255,203/255,203/255)) 
gpuAnillo7 = createGPUShape(createColorRing(500,1.68,1.7,183/255,183/255,183/255)) 
gpuAnillo8 = createGPUShape(createColorRing(500,1.71,1.73,173/255,173/255,173/255)) 
gpuAnillo9 = createGPUShape(createColorRing(500,1.74,1.76,163/255,163/255,163/255)) 
gpuAnillo10 = createGPUShape(createColorRing(500,1.77,1.79,153/255,153/255,153/255))

gpuAnilloUrano = createGPUShape(createColorRing(500,1.2,1.4,91/255,230/255,246/255))

#grafo de escena:
solarSystem = createSystem(pipeline)


#esta parte del codigo controla la vista:

camera_theta = np.pi/4

# Setting up the view transform

cam_radius = 10
cam_x = cam_radius * np.sin(camera_theta)
cam_y = cam_radius * np.cos(camera_theta)
cam_z = cam_radius

viewPos = np.array([cam_x, cam_y, cam_z])

view = tr.lookAt(viewPos, np.array([0, 0, 0]), np.array([0, 0, 1]))

# Setting up the projection transform

projection = tr.ortho(-8, 8, -8, 8, 0.1, 100)

perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

# glfw will swap buffers as soon as possible
glfw.swap_interval(0)

#animacion:
while not glfw.window_should_close(window):

    # Measuring performance
    perfMonitor.update(glfw.get_time())
    glfw.set_window_title(window, title + str(perfMonitor))

    # Using GLFW to check for input events
    glfw.poll_events()

    #vista y proyeccion:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    # Clearing the screen in both, color and depth
    glClear(GL_COLOR_BUFFER_BIT)

    theta = glfw.get_time() #tiempo
    model = tr.uniformScale(13.5)

    glUseProgram(texturePipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texturePipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    glUniformMatrix4fv(glGetUniformLocation(texturePipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniformMatrix4fv(glGetUniformLocation(texturePipeline.shaderProgram, "model"), 1, GL_TRUE, model)

    #dibujar fondo:
    glUseProgram(texturePipeline.shaderProgram)
    texturePipeline.drawCall(gpuBackground)

    glUseProgram(pipeline.shaderProgram)

    #dibujamos los planetas y sus anillos:

    #Mercurio:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1), #mover a otro plano
        tr.rotationY(1.5*theta), #translacion alrededor del sol
        tr.rotationX(0.2*theta), #rotacion del planeta
        tr.translate(3, 0, 0), #moverlo del centro
        tr.uniformScale(0.06)])) #escalamiento 
    pipeline.drawCall(gpuMercurio)

    #Venus:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*0.99),
        tr.rotationY(1.425*theta),
        tr.rotationX(0.1*theta),
        tr.translate(3.2, 0, 0),
        tr.uniformScale(0.19)]))
    pipeline.drawCall(gpuVenus)

    #Marte:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.05),
        tr.rotationY(1.3*theta),
        tr.rotationX(0.99*theta),
        tr.translate(4.2, 0, 0),
        tr.uniformScale(0.11)]))
    pipeline.drawCall(gpuMarte)

    #Jupiter:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.07),
        tr.rotationY(1.25*theta),
        tr.rotationX(3*theta),
        tr.translate(6, 0, 0),
        tr.uniformScale(0.64)]))
    pipeline.drawCall(gpuJupiter)

    #Saturno:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.rotationX(2.9*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuSaturno)

    #anillos Saturno:

    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo1)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo2)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo3)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo4)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo5)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo6)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo7)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo8)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo9)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.01),
        tr.rotationY(1.2*theta),
        tr.translate(7.5, 0, 0),
        tr.uniformScale(0.58)]))
    pipeline.drawCall(gpuAnillo10)

    #Urano:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.02),
        tr.rotationY(1.15*theta),
        tr.rotationX(2*theta),
        tr.translate(8.0, 0, 0),
        tr.uniformScale(0.41)]))
    pipeline.drawCall(gpuUrano)

    #anillo Urano:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.02),
        tr.rotationY(1.15*theta),
        tr.translate(8.0, 0, 0),
        tr.uniformScale(0.41)]))
    pipeline.drawCall(gpuAnilloUrano)

    #Neptuno:
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.matmul([
        tr.rotationZ(np.pi*1.03),
        tr.rotationY(1.1*theta),
        tr.rotationX(2.1*theta),
        tr.translate(8.6, 0, 0),
        tr.uniformScale(0.4)]))
    pipeline.drawCall(gpuNeptuno)

    #transformaciones grafo de escena:
    
    sunRot = sg.findNode(solarSystem, "sunRotation")
    sunRot.transform = tr.rotationY(0.5*theta)

    earthRot = sg.findNode(solarSystem, "earthRotation")
    earthRot.transform = tr.rotationY(theta)
    
    moonRot = sg.findNode(solarSystem, "moonRotation")
    moonRot.transform = tr.rotationY(0.5*theta)

    moonSystem = sg.findNode(solarSystem, "moonSystem")
    moonSystem.transform = tr.rotationY(3.5*theta)
    
    earthSystem = sg.findNode(solarSystem, "earthSystem")
    earthSystem.transform = tr.rotationY(1.35*theta)
    
    sg.drawSceneGraphNode(solarSystem, pipeline, "model")


    # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
    glfw.swap_buffers(window)

# freeing GPU memory
gpuBackground.clear()
gpuMercurio.clear()
gpuVenus.clear()
earthSystem.clear()
gpuMarte.clear()
gpuVenus.clear()
gpuSaturno.clear()
gpuUrano.clear()
gpuNeptuno.clear()
gpuAnillo1.clear()
gpuAnillo2.clear()
gpuAnillo3.clear()
gpuAnillo4.clear()
gpuAnillo5.clear()
gpuAnillo6.clear()
gpuAnillo7.clear()
gpuAnillo8.clear()
gpuAnillo9.clear()
gpuAnillo10.clear()
gpuAnilloUrano.clear()
solarSystem.clear()

glfw.terminate()