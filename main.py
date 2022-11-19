from carla_lib import *
#params
width = 768
height = 384*2

# pygame 
import pygame
world = None

#tensorflow 
#load model 
model = tf.keras.models.load_model("unet", {"SparseCategoricalFocalLoss":SparseCategoricalFocalLoss(gamma=2)})

try:
    #pygame setup 
    pygame.init()
    clock = pygame.time.Clock()
    display = pygame.display.set_mode((width,height))
    pygame.display.set_caption("game")

    #connect to server 
    client = carla.Client("localhost", 2000)
    client.set_timeout(4.0)

    #get world
    world = World(client.get_world())
    #spawn ego vehicle
    world.spawn_ego_car(2)
    
    #spawn cameras
    box = world.vehicle.bounding_box
    spawn_points = [carla.Transform(carla.Location(x= box.location.x - 4, y= box.location.y , z = box.location.z ), carla.Rotation(0,0,0)),
                    carla.Transform(carla.Location(x= box.location.x+box.extent.x, y= box.location.y, z=box.location.z + 0.5 * box.extent.z), carla.Rotation(0, 0, 0))
                    ]
    
    #where to render the cameras images on the pygame display
    render_pos = [(0,0), (0, 384)]
    
    world.spawn_camera(spawn_points[0], render_pos[0], (768, 384), attach_to=world.vehicle, type = 0, model=None)
    world.spawn_camera(spawn_points[1], render_pos[1], (768, 384), attach_to=world.vehicle, type = 0, model=model)

    #spawn 15 radom npc cars
    for i in range(15):
        world.spawn_npc()

    world.world.tick()
    vehicles = world.world.get_actors().filter('vehicle.*')
    
    #get vehicle controler
    controller = KeyboardControl(world)

    #mains cameras potisions
    box = world.vehicle.bounding_box
    mcp = [(carla.Location(x = -10 , y= 0 , z = 2), carla.Rotation(-20,0,0)), (carla.Location(x = 0 , y= -5 , z = 2), carla.Rotation(-20,90,0)), (carla.Location(x = 5 , y= 2 , z = 2), carla.Rotation( -20,200,0)), (carla.Location(x= box.location.x+box.extent.x, y= box.location.y - box.extent.y, z=box.location.z + 0.5 * box.extent.z), carla.Rotation(0,0,0)) ]
    mcp_index = 0

    #vector space 
    vector_space = VectorSpace()
    #game loop 
    while True:
        #check events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    world.sensor_list[0].set_camera_pos(mcp[mcp_index][0], mcp[mcp_index][1])
                    mcp_index+=1
                    mcp_index = mcp_index%len(mcp)
                if event.key == pygame.K_r:
                    if controller.reverse == False:
                        controller.reverse = True
                    else:
                        controller.reverse = False
                if event.key == pygame.K_l:
                    world.save_cameras_frames()

        controller.parse_events(clock)
        display.fill((0, 0, 0))
        world.render(display)
        
        pygame.display.flip()
        world.world.tick()
        clock.tick(30)

finally:
    if world is not None:
        world.destory_actors()
    print("done")
