# Space Efficient Context Encoding for Non-Task-Oriented Dialogue Generation with Graph Attention Transformer #

This is the repository for the above mentioned paper. It can be found here: [link](https://aclanthology.org/2021.acl-long.546/)


### Knowledge Graphs ###

We provide the generated knowledge graphs (depth 0 & 1) for the 

* KOMODIS dataset ([Link](https://github.com/fabiangal/komodis-dataset) to official repository)
* and OpenDialKG dataset ([Link](https://github.com/facebookresearch/opendialkg) to official repository)

in *data/knowledge_graphs/*. Please download the datasets in *data/dataset/*. It will not work
with the version from the official repositories. Please unzip all files before running scripts.

If you are interested in the graphs with depth > 1, please contact us. 
We will send you corresponding download links.

If you use the OpenDialKG dataset make sure to cite the authors correctly as well 
(citation information can be found in the linked dataset repository above).


### Data Processing and Model ###

To run the graph pre-processing described in Chapter 4.2 please use the following command:

````
python process_graphs.py --dataset komodis --depth 0 --encoding series
````

To run a training please use:

````
python train.py --dataset komodis --depth 0 --encoding series
````

### Reference ###

````
@inproceedings{galetzka-etal-2021-space,
    title = "Space Efficient Context Encoding for Non-Task-Oriented Dialogue Generation with Graph Attention Transformer",
    author = "Galetzka, Fabian  and
      Rose, Jewgeni  and
      Schlangen, David  and
      Lehmann, Jens",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.546",
    doi = "10.18653/v1/2021.acl-long.546",
    pages = "7028--7041"
}
````

### License ###

MIT License

Copyright (c) 2021 Fabian Galetzka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
