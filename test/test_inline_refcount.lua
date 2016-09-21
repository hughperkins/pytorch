require 'torch'

torch.setdefaulttensortype('torch.FloatTensor')

local TestInlineRefCount = torch.class('TestInlineRefCount')

function TestInlineRefCount:__init()
    print("<TestInlineRefCount> initialization")
end

function TestInlineRefCount:write()
    if self.v then
        print("<TestInlineRefCount> This is the value:")
        print(self.v)
    else
        print("<TestInlineRefCount> Value not defined")
    end
end

function TestInlineRefCount:set(vin)
    self.v = vin
end
