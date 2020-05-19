def draw_line(draw_ob, point1, point2, color, line_width):
    '''
    draw line on image
    '''

    if not point1 is None and point2 is not None:
        draw_ob.line([point1, point2], fill=color, width=line_width)


def draw_dot(draw_ob, point, color, radius):
    '''
    draws dot (filled circle) on image
    '''

    if point is not None:
        xy = [
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius
        ]
        draw_ob.ellipse(xy, fill=color, outline=color)


def draw_cube(draw_ob, points, color=(255, 0, 0), line_width=2,
              draw_dots=False, draw_x=False):
    '''
    draw cube with a thick solid line across
    the front top edge and an X on the top face
    '''

    # draw front
    draw_line(draw_ob, points[0], points[1], color, line_width)
    draw_line(draw_ob, points[1], points[2], color, line_width)
    draw_line(draw_ob, points[3], points[2], color, line_width)
    draw_line(draw_ob, points[3], points[0], color, line_width)

    # draw back
    draw_line(draw_ob, points[4], points[5], color, line_width)
    draw_line(draw_ob, points[6], points[5], color, line_width)
    draw_line(draw_ob, points[6], points[7], color, line_width)
    draw_line(draw_ob, points[4], points[7], color, line_width)

    # draw sides
    draw_line(draw_ob, points[0], points[4], color, line_width)
    draw_line(draw_ob, points[7], points[3], color, line_width)
    draw_line(draw_ob, points[5], points[1], color, line_width)
    draw_line(draw_ob, points[2], points[6], color, line_width)

    # draw dots
    if draw_dots:
        draw_dot(draw_ob, points[0], pointColor=color, radius=4)
        draw_dot(draw_ob, points[1], pointColor=color, radius=4)

    # draw x on the top
    if draw_x:
        draw_line(draw_ob, points[0], points[5], color, line_width)
        draw_line(draw_ob, points[1], points[4], color, line_width)
