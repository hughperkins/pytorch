require 'torch'

local ThrowsError = torch.class('ThrowsError')

function ThrowsError:__init()
end

function ThrowsError:go()
  a = nil
  a.foo()
end

local ThrowsErrorOnInit = torch.class('ThrowsErrorOnInit')

function ThrowsErrorOnInit:__init()
  b = nil
  b.foo()
end

