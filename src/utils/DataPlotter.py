import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
from sklearn.manifold import TSNE
import time
import itertools
import numpy as np

import hdbscan
import matplotlib.pyplot as plt

import DataLoader
import Config


def plot_clustering(model, path):
	#TSNE
	model_tsne = load_TSNE(model)

	print "\nHDBSCAN Started..."
	print model_tsne.shape

	clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=3, alpha=0.2).fit(model_tsne)

	print "HDBSCAN done..."

	print "\nPlotly Started..."

	num_cluster = len(set(clusterer.labels_))
	print "num_cluster", num_cluster

	#labels
	labels = []
	for label in model.vocab:
		labels.append(label)

	#legend_labels
	clusters = map(str, clusterer.labels_)
	
	cluster2color = {
		'0' :  sns.xkcd_rgb["purple"],
		'1' : sns.xkcd_rgb["forest green"],
		'2' : sns.xkcd_rgb["light pink"],
		'3' : sns.xkcd_rgb["mustard yellow"],
		'4' : sns.xkcd_rgb["orange"],
		'5' : sns.xkcd_rgb["magenta"],
		'6' : sns.xkcd_rgb["purple"],
		'7' : sns.xkcd_rgb["pink"],
		'8' : sns.xkcd_rgb["deep blue"],
		'9' : sns.xkcd_rgb["sky blue"],
		'10' : sns.xkcd_rgb["olive"],
		'11' : sns.xkcd_rgb["red"],
		'12' : sns.xkcd_rgb["yellow"],
		'13' : sns.xkcd_rgb["lime"],
		'14' : sns.xkcd_rgb["salmon"],
		'15' : sns.xkcd_rgb["navy"],
		'16' : sns.xkcd_rgb["beige"],
		'17' : sns.xkcd_rgb["mint"],
		'18' : sns.xkcd_rgb["crimson"],
		'19' : sns.xkcd_rgb["azure"],
		'20' : sns.xkcd_rgb["lemon"],
		'-1' : sns.xkcd_rgb["grey"],
	}

	cluster_order = [
		'0',
		'1',
		'2',
		'3',
		'4',
		'5',
		'6',
		'7',
		'8',
		'9',
		'10',
		'11',
		'12',
		'13',
		'14',
		'15',
		'16',
		'17',
		'18',
		'19',
		'20',
		'-1',
	]

	make_plot_with_labels_legends(name=path,
		  points=model_tsne, 
		  labels=labels, 
		  legend_labels=clusters, 
		  legend_order=cluster_order, 
		  legend_label_to_color=cluster2color, 
		  pretty_legend_label=pretty_category,
		  publish=False)
	
	print "Plotly Done..."

	



def plot_pipeline(model, path, withLegends=False):

	#TSNE
	model_tsne = load_TSNE(model)

	#Label Load
	labels = []
	for label in model.vocab:
		labels.append(label)

	#Legend Load
	if withLegends:
		dl = DataLoader.DataLoader()
		ingredients = dl.load_ingredients(Config.path_ingr_info)

		categories = []
		for label in labels:
			categories.append(dl.ingredient_to_category(label,ingredients))
		
		categories_color = list(set(categories))

		category2color = {
			'plant' :  sns.xkcd_rgb["purple"],
			'flower' : sns.xkcd_rgb["forest green"],
			'meat' : sns.xkcd_rgb["light pink"],
			'nut/seed/pulse' : sns.xkcd_rgb["mustard yellow"],
			'herb' : sns.xkcd_rgb["orange"],
			'alcoholic beverage' : sns.xkcd_rgb["magenta"],
			'plant derivative' : sns.xkcd_rgb["purple"],
			'fruit' : sns.xkcd_rgb["blue"],
			'dairy' : sns.xkcd_rgb["deep blue"],
			'cereal/crop' : sns.xkcd_rgb["sky blue"],
			'vegetable' : sns.xkcd_rgb["olive"],
			'animal product' : sns.xkcd_rgb["red"],
			'fish/seafood' : sns.xkcd_rgb["yellow"],
			'spice' : sns.xkcd_rgb["black"],
		}

		category_order = [
			'plant',
			'flower',
			'meat',
			'nut/seed/pulse',
			'herb',
			'alcoholic beverage',
			'plant derivative',
			'fruit',
			'dairy',
			'cereal/crop',
			'vegetable',
			'animal product',
			'fish/seafood',
			'spice',
		]

		make_plot_with_labels_legends(name=path,
		  points=model_tsne, 
		  labels=labels, 
		  legend_labels=categories, 
		  legend_order=category_order, 
		  legend_label_to_color=category2color, 
		  pretty_legend_label=pretty_category,
		  publish=False)

	else:
		make_plot_only_labels(name=path+'/compound2vec_171018',
				  points=model_tsne, 
				  labels=labels, 
				  publish=False)

"""
TSNE of Doc2Vec

"""
def load_TSNE(model, dim=2):
	print "\nt-SNE Started... "
	time_start = time.time()

	X = []
	for x in model.vocab:
		X.append(model.word_vec(x))
	tsne = TSNE(n_components=dim)
	X_tsne = tsne.fit_transform(X)

	print "t-SNE done!"
	print "Time elapsed: {} seconds".format(time.time()-time_start)

	return X_tsne

"""
Load functions for plotting a graph
"""

flatten = lambda l: [item for sublist in l for item in sublist]

# Prettify ingredients
pretty_food = lambda s: ' '.join(s.split('_')).capitalize().lstrip()
# Prettify cuisine names
pretty_category = lambda s: ''.join(map(lambda x: x if x.islower() else " "+x, s)).lstrip()

"""
Plot Points with Labels
"""
def make_plot_only_labels(name, points, labels, publish):
	traces = []
	traces.append(go.Scattergl(
			x = points[:, 0],
			y = points[:, 1],
			mode = 'markers',
			marker = dict(
				color = sns.xkcd_rgb["black"],
				size = 8,
				opacity = 0.6,
				#line = dict(width = 1)
			),
			text = labels,
			hoverinfo = 'text',
		)
		)
				  
	layout = go.Layout(
		xaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=False,
			showline=False,
			autotick=True,
			ticks='',
			showticklabels=False
		),
		yaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=False,
			showline=False,
			autotick=True,
			ticks='',
			showticklabels=False
		)
		)
				  
	fig = go.Figure(data=traces, layout=layout)
	if publish:
		plotter = py.iplot
	else:
		plotter = offline.plot
	plotter(fig, filename=name + '.html')

"""
Plot Points with Labels and Legends
"""

def make_plot_with_labels_legends(name, points, labels, legend_labels, legend_order, legend_label_to_color, pretty_legend_label, publish):
	lst = zip(points, labels, legend_labels)
	full = sorted(lst, key=lambda x: x[2])
	traces = []
	for legend_label, group in itertools.groupby(full, lambda x: x[2]):
		group_points = []
		group_labels = []
		for tup in group:
			point, label, _ = tup
			group_points.append(point)
			group_labels.append(label)
		group_points = np.stack(group_points)
		traces.append(go.Scattergl(
			x = group_points[:, 0],
			y = group_points[:, 1],
			mode = 'markers',
			marker = dict(
				color = legend_label_to_color[legend_label],
				size = 8,
				opacity = 0.6,
				#line = dict(width = 1)
			),
			text = ['{} ({})'.format(label, pretty_legend_label(legend_label)) for label in group_labels],
			hoverinfo = 'text',
			name = legend_label
		)
		)
	# order the legend
	ordered = [[trace for trace in traces if trace.name == lab] for lab in legend_order]
	traces_ordered = flatten(ordered)
	def _set_name(trace):
		trace.name = pretty_legend_label(trace.name)
		return trace
	traces_ordered = list(map(_set_name, traces_ordered))
	
	"""
	annotations = []
	for index in range(50):
		new_dict = dict(
				x=points[:, 0][index],
				y=points[:, 1][index],
				xref='x',
				yref='y',
				text=labels[index],
				showarrow=True,
				arrowhead=7,
				ax=0,
				ay=-10
			)
		annotations.append(new_dict)
	"""
	
	layout = go.Layout(
		xaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=True,
			showline=True,
			autotick=True,
			ticks='',
			showticklabels=False
		),
		yaxis=dict(
			autorange=True,
			showgrid=False,
			zeroline=True,
			showline=True,
			autotick=True,
			ticks='',
			showticklabels=False
		),
		#annotations=annotations
	)
	fig = go.Figure(data=traces_ordered, layout=layout)
	if publish:
		plotter = py.iplot
	else:
		plotter = offline.plot
	plotter(fig, filename=name + '.html')