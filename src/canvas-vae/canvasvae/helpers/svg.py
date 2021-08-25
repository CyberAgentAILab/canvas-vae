import logging
import xml.etree.ElementTree as ET

NS = {
    'svg': 'http://www.w3.org/2000/svg',
    'xlink': 'http://www.w3.org/1999/xlink',
}
ET.register_namespace('', NS['svg'])
ET.register_namespace('xlink', NS['xlink'])

logger = logging.getLogger(__name__)

# DUMMY_TEXT = '''
# Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
# incididunt ut labore et dolore magna aliqua.
# '''
DUMMY_TEXT = '''
TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT
'''


class SVGBuilder(object):
    """
    Utility to generate SVG for visualization.

    Usage::

        dataspec = DataSpec(...)
        dataset = dataspec.make_dataset('val')
        example = next(iter(dataset))

        # Manual colormap.
        builder = SVGBuilder(
            'type',
            colormap={
                '': 'none',
                'svgElement': 'blue',
                'textElement': 'red',
                'imageElement': 'green',
                'maskElement': 'cyan',
                'coloredBackground': 'magenta',
                'videoElement': 'yellow',
            },
            max_width=144,
        )
        for item in dataspec.unbatch(example):
            svg = builder(item)

        # Auto colormap by preprocessor.
        builder = SVGBuilder(
            'component',
            preprocessor=dataspec.preprocessor,
            max_width=144,
        )
        for item in dataspec.unbatch(example):
            svg = builder(item)

    """
    def __init__(
        self,
        key=None,
        preprocessor=None,
        colormap=None,
        canvas_width=None,
        canvas_height=None,
        max_width=None,
        max_height=None,
        opacity=0.5,
        image_db=None,
        render_text=False,
        **kwargs,
    ):
        assert key
        self._key = key
        self._canvas_width = canvas_width or 256
        self._canvas_height = canvas_height or 256
        self._max_width = max_width
        self._max_height = max_height
        self._opacity = opacity
        self._render_text = render_text
        assert preprocessor or colormap
        if preprocessor is None or key == 'color':
            self._colormap = colormap
        else:
            vocabulary = preprocessor[key].get_vocabulary()
            self._colormap = self._make_colormap(vocabulary, colormap)
        self._image_db = image_db
        self._image_condition_key = 'type'
        self._image_types = {'svgElement', 'imageElement', 'maskElement'}
        self._image_key = 'image_embedding'

    def __call__(self, document):
        canvas_width, canvas_height = self.compute_canvas_size(document)
        root = ET.Element(
            ET.QName(NS['svg'], 'svg'), {
                'width': str(canvas_width),
                'height': str(canvas_height),
                'viewBox': '0 0 1 1',
                'style': 'background-color: #EEE',
                'preserveAspectRatio': 'none',
            })
        for element in document['elements']:
            if self._key == 'color':
                fill = 'rgb(%g,%g,%g)' % tuple(map(int, element['color']))
            else:
                fill = self._colormap.get(element[self._key], 'none')

            image_url = ''
            if self._image_db and element.get(
                    self._image_condition_key) in self._image_types:
                image_feature = element.get(self._image_key)
                image_url = self._image_db.search(image_feature)

            if image_url:
                node = self._make_image(root, element, image_url)
            elif self._render_text and element.get('type') == 'textElement':
                node = self._make_text_element(root, element, fill,
                                               canvas_width)
            else:
                node = self._make_rect(root, element, fill)

            title = ET.SubElement(node, ET.QName(NS['svg'], 'title'))
            title.text = str({
                k: v
                for k, v in element.items()
                if not (self._image_db and k == self._image_key)
            })
        return ET.tostring(root).decode('utf-8')

    def compute_canvas_size(self, document):
        canvas_width = document.get('canvas_width', self._canvas_width)
        canvas_height = document.get('canvas_height', self._canvas_height)
        scale = 1.0
        if self._max_width is not None:
            scale = min(self._max_width / canvas_width, scale)
        if self._max_height is not None:
            scale = min(self._max_height / canvas_height, scale)
        return canvas_width * scale, canvas_height * scale

    def _make_colormap(self, vocabulary, colormap=None):
        """
        Generate a colormap for the specified vocabulary list.
        """
        from matplotlib import cm

        vocab_size = len(vocabulary)
        cmap = cm.get_cmap(colormap or 'tab20', vocab_size)
        return {
            label: 'rgb(%g,%g,%g)' % tuple(int(x * 255) for x in c[:3])
            for label, c in zip(vocabulary, cmap(range(vocab_size)))
        }

    def _make_text_element(self, parent, element, fill, canvas_width):
        opacity = element.get('opacity', 1.0)
        ET.SubElement(
            parent,
            ET.QName(NS['svg'], 'rect'),
            {
                'x': str(element['left']),
                'y': str(element['top']),
                'width': str(element['width']),
                'height': str(element['height']),
                # 'stroke': str(fill),
                # 'stroke-width': str(1 / float(canvas_width)),
                'fill': str(fill),
                'opacity': str(opacity * .3),
            })
        # Clip-box for the text element.
        _svg = ET.SubElement(
            parent, ET.QName(NS['svg'], 'svg'), {
                'x': str(element['left']),
                'y': str(element['top']),
                'width': str(element['width']),
                'height': str(element['height']),
                'overflow': 'hidden',
            })
        node = ET.SubElement(
            _svg, ET.QName(NS['svg'], 'text'), {
                'x': str(0),
                'y': str(element['height']),
                'opacity': str(opacity * .7),
                'font-size': str(element['height']),
                'fill': str(fill),
                'style': 'vertical-align:top;font-stretch:condensed;',
            })
        node.text = DUMMY_TEXT
        return node

    def _make_image(self, parent, element, image_url):
        return ET.SubElement(
            parent, ET.QName(NS['svg'], 'image'), {
                'x': str(element['left']),
                'y': str(element['top']),
                'width': str(element['width']),
                'height': str(element['height']),
                ET.QName(NS['xlink'], 'href'): image_url,
                'opacity': str(element.get('opacity', 1.0)),
                'preserveAspectRatio': 'none',
            })

    def _make_rect(self, parent, element, fill):
        return ET.SubElement(
            parent, ET.QName(NS['svg'], 'rect'), {
                'x': str(element['left']),
                'y': str(element['top']),
                'width': str(element['width']),
                'height': str(element['height']),
                'fill': str(fill),
                'opacity': str(element.get('opacity', 1.0) * self._opacity),
            })
