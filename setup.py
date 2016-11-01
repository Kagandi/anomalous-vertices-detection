from distutils.core import setup

setup(name='anomalous-vertices-detection',
      version='0.6',
      description='Anomalous vertices detection method',
      author='Dima Kagan',
      author_email='kagandi@post.bgu.ac.il',
      url='https://github.com/Kagandi/anomalous-vertices-detection',
      packages=['anomalous_vertices_detection',
                'anomalous_vertices_detection',
                'anomalous_vertices_detection.configs',
                'anomalous_vertices_detection.graphs',
                'anomalous_vertices_detection.learners',
                'anomalous_vertices_detection.samplers',
                'anomalous_vertices_detection.utils'],
      )
