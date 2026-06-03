from benchmark import build_scene, GridBenchmark, RobotFootprint
scene = [
    '####################',
    '#S       #        #',
    '#        #        #',
    '#        #        #',
    '#        #        #',
    '#        #       G#',
    '####################',
]
grid, start, goal = build_scene(scene)
robot = RobotFootprint(body_width=1, body_length=2, leg_margin=1, sensor_margin=0, safe_margin=1)
bench = GridBenchmark(grid, start, goal, robot)
print('start', start, 'goal', goal, 'inflation', robot.inflation_radius, 'eff_width', robot.eff_width)
print('start free', bench.is_free(*start), 'goal free', bench.is_free(*goal))
for row in bench.inflated:
    print(''.join('#' if cell else '.' for cell in row))
