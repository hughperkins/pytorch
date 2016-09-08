require 'torch'
require 'nn'

local TestCallLua = torch.class('TestCallLua')

function TestCallLua:__init(someName)
  print('TestCallLua:__init(', someName, ')')
  self.someName = someName
end

function TestCallLua:getName()
  return self.someName
end

function TestCallLua:getOut(inTensor, outSize, kernelSize)
  local inSize = inTensor:size(3)
  local m = nn.TemporalConvolution(inSize, outSize, kernelSize)
  m:float()
  local out = m:forward(inTensor)
  print('out from lua', out)
  return out
end

function TestCallLua:addOne(inTensor)
  local outTensor = inTensor + 3
  return outTensor
end

function TestCallLua:printTable(sometable, somestring, table2)
  for k, v in pairs(sometable) do
    print('TestCallLua:printTable ', k, v)
  end
  print('somestring', somestring)
  for k, v in pairs(table2) do
    print('TestCallLua table2 ', k, v)
  end
  return {bear='happy', result=12.345, foo='bar'}
end
