require 'torch'
require 'nn'

local TorchModel = torch.class('TorchModel')

function TorchModel:__init(backend, imageSize, numClasses)
  self:buildModel(backend, imageSize, numClasses)
  self.imageSize = imageSize
  self.numClasses = numClasses
  self.backend = backend
end

function TorchModel:buildModel(backend, imageSize, numClasses)
  self.net = nn.Sequential()
  local net = self.net

  -- simple mnist network
  -- ====================

  -- params of convolutionmm are:
  --   (inFeatures, outFeatures, kernelWidth, kernelHeight, horizStride, vertStride,
  --    horizPadding, vertPadding)
  -- ( https://github.com/torch/nn/blob/master/SpatialConvolutionMM.lua#L3 )
  -- so, this maps from 1 feature plan to 16 feature planes.
  -- it's a 5x5 convolution, stride 1, with padding
  net:add(nn.SpatialConvolutionMM(1, 16, 5, 5, 1, 1, 2, 2))
  net:add(nn.ReLU())

  -- params are: (pooling width, pooling height, horizontal stride, vertical stride)
  -- https://github.com/torch/nn/blob/master/SpatialMaxPooling.lua
  net:add(nn.SpatialMaxPooling(3, 3, 3, 3))

  -- change from 16 to 32 feature planes.  This is a 3x3 convolution, with padding, stride 1
  net:add(nn.SpatialConvolutionMM(16, 32, 3, 3, 1, 1, 1, 1))
  net:add(nn.ReLU())
  net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- reshape from being 4-dimensional tensor (example number, feature plane, width, height), to
  -- being 2 dimensional: (example number, feature plane * width * height)
  net:add(nn.Reshape(32 * 4 * 4))

  -- fully connected layer, with 150 output neurons
  net:add(nn.Linear(32 * 4 * 4, 150))
  net:add(nn.Tanh())

  -- fully connected layer, with numClasses output neurons
  net:add(nn.Linear(150, numClasses))

  -- softmax.  Actually, this gives the log of the softmax output
  -- our loss criterion will correspondingly expect the log of the softmax output as input, see below
  net:add(nn.LogSoftMax())

  self.crit = nn.ClassNLLCriterion()  -- this is the loss function for labelled examples, given the network
                                      -- outputs the log soft max

  if backend == 'cuda' then
    require 'cutorch'
    require 'cunn'
    self.net:cuda()
    self.crit:cuda()
  elseif backend == 'cl' then
    require 'cltorch'
    require 'clnn'
    self.net:cl()
    self.crit:cl()
  else
    self.net:float()  -- default is double
    self.crit:float()
  end

  print('self.net', self.net)
  print('self.crit', self.crit)
  print('network created')
end

function TorchModel:processData(input, labels)
  -- copies into self.batchInput, converting into cl or cuda as necessary
  -- labels is optional

  local batchSize = input:size(1)
  local backend = self.backend
  if backend == 'cpu' then
    self.batchInput = input  -- on cpu, just copy the reference
  else
    self.batchInput = self.batchInput or input:clone()
    if backend == 'cuda' then
      if torch.type(self.batchInput) ~= 'torch.CudaTensor' then
        self.batchInput = self.batchInput:cuda()
      end
    elseif backend == 'cl' then
      if torch.type(self.batchInput) ~= 'torch.ClTensor' then
        self.batchInput = self.batchInput:cl()
      end
    end
    -- copy from cpu memory to gpu memory
    self.batchInput:resize(batchSize, 1, self.imageSize, self.imageSize)
    self.batchInput:copy(input)
  end
  self.batchInput:resize(batchSize, 1, self.imageSize, self.imageSize)
  if labels == nil then
    return
  end

  if backend == 'cpu' then
    self.batchLabels = labels
  else
    self.batchLabels = self.batchLabels or labels:clone()
    if backend == 'cuda' then
      if torch.type(self.batchLabels) ~= 'torch.CudaTensor' then
        self.batchLabels = self.batchLabels:cuda()
      end
    elseif backend == 'cl' then
      if torch.type(self.batchLabels) ~= 'torch.ClTensor' then
        self.batchLabels = self.batchLabels:cl()
      end
    end
    -- copy from cpu memory to gpu memory
    self.batchLabels:resize(batchSize)
    self.batchLabels:copy(labels)
  end
  self.batchLabels:resize(batchSize)
end

function TorchModel:trainBatch(learningRate, input, labels)
  -- assume data arrive as torch.FloatTensors

  local batchSize = labels:size(1)
--  print('batchSize', batchSize)

  -- copy to cuda or cl tensors, if we need to
  self:processData(input, labels)

  self.net:zeroGradParameters()
  local output = self.net:forward(self.batchInput)
  local _, prediction = output:max(2)
--  print('prediction', prediction)
  local numRight = labels:int():eq(prediction:int()):sum()
--  print('numRight', numRight)
  local loss = self.crit:forward(output, self.batchLabels)
  local gradOutput = self.crit:backward(output, self.batchLabels)
  self.net:backward(self.batchInput, gradOutput)
  self.net:updateParameters(learningRate)
  return {loss=loss, numRight=numRight}  -- you can return a table, it will become a python dictionary
end

function TorchModel:predict(input)
  -- assume batched
  self:processData(input)
  local batchSize = input:size(1)
  local output = self.net:forward(self.batchInput)
  local _, prediction = output:max(2)
  local predictionAsBytes = prediction:byte()
  return predictionAsBytes
end

