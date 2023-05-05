

def init():

    global device_level_failure
    global failed_device_index
    global num_device
    global packet_size
    global packet_loss_percentage
    global accuracy


    device_level_failure=False
    failed_device_index=1
    num_device=1
    packet_size=64
    packet_loss_percentage=0.1
    accuracy = 0