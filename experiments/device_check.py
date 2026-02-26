import openvino as ov 
import openvino.properties as props

core = ov.Core()

devices = core.available_devices

for device in devices: 
    device_name = core.get_property(device, props.device.full_name)
    print(f"{device}: {device_name}")
