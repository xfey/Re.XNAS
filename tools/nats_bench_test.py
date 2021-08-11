from nats_bench import create

api = create(None, 'sss', False, True)

info = api.get_more_info(1234, 'cifar10')
print(info)
