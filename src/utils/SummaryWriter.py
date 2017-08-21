import tensorflow as tf


class SummaryWriter(tf.summary.FileWriter):
    def __init__(self, log_dir, session, summaries,
                 graph=None, max_queue=10, flush_secs=120, graph_def=None, filename_suffix=None):
        super(SummaryWriter, self).__init__(log_dir, graph, max_queue, flush_secs, graph_def, filename_suffix)
        self.session = session
        self.summaries = summaries

    def update(self):
        summaries = self.session.run(self.summaries)
        self.add_summary(summaries)
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
