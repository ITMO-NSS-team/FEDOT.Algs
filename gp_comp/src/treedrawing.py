import os

from PIL import Image, ImageDraw


class TreeDrawing:

    def _getwidth(self, node):
        if not node.nodes_from:
            return 1
        else:
            result = 0
            for i in range(0, len(node.nodes_from)):
                result += self._getwidth(node.nodes_from[i])
            return result

    def _draw_node(self, node, draw, x, y):
        if node.nodes_from:
            all_width = 0
            for c in node.nodes_from:
                all_width += self._getwidth(c) * 100
            left = x - all_width / 2
            # draw the function name
            draw.text((x - 10, y - 10), node.eval_strategy.model.__class__.__name__, (0, 0, 0))

            # draw the children
            for c in node.nodes_from:
                wide = self._getwidth(c) * 100
                draw.line((x, y, left + wide / 2, y + 100), fill=(255, 0, 0))
                self._draw_node(c, draw, left + wide / 2, y + 100)
                left = left + wide
        else:
            draw.text((x - 5, y), str(node.eval_strategy.model.__class__.__name__), (0, 0, 0))

    def draw_branch(self, node, jpeg="tree.png"):
        if not os.path.isdir(f'HistoryFiles'):
            os.mkdir(f'HistoryFiles')
        if not os.path.isdir(f'HistoryFiles/Trees'):
            os.mkdir(f'HistoryFiles/Trees')

        w = self._getwidth(node) * 100
        if not node.nodes_from:
            h = 100 + 120
        else:
            h = node.get_depth_down() * 100 + 120

        img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        self._draw_node(node, draw, w / 2, 20)
        img.save(f'HistoryFiles/Trees/{jpeg}', 'PNG')
