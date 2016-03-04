require 'torch'
require 'nn'

local Luabit = torch.class('Luabit')

function Luabit:__init(someName)
  print('Luabit:__init(', someName, ')')
  self.someName = someName
end

function Luabit:getName()
  return self.someName
end

function Luabit:getOut(inTensor, outSize, kernelSize)
  local inSize = inTensor:size(3)
  local m = nn.TemporalConvolution(inSize, outSize, kernelSize)
  m:float()
  local out = m:forward(inTensor)
  print('out from lua', out)
  return out
end

function Luabit:printTable(sometable)
  for k, v in pairs(sometable) do
    print('Luabit:printTable ', k, v)
  end
end

