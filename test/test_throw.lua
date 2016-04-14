require 'torch'

local ThrowsError = torch.class('ThrowsError')

function ThrowsError:__init()
end

function ThrowsError:go()
  a = nil
  a.foo()
end

function mysub()
  error('error in mysub')
end

function ThrowsError:insub_anteater()
  mysub()
end

local ThrowsErrorOnInit = torch.class('ThrowsErrorOnInit')

function ThrowsErrorOnInit:__init()
  b = nil
  b.foo()
end

