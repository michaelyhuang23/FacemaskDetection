Ô×*
Í£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18Ûò"

rcnn_1/conv2d_196/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namercnn_1/conv2d_196/kernel

,rcnn_1/conv2d_196/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_196/kernel*(
_output_shapes
:*
dtype0

rcnn_1/conv2d_196/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namercnn_1/conv2d_196/bias
~
*rcnn_1/conv2d_196/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_196/bias*
_output_shapes	
:*
dtype0
¡
$rcnn_1/batch_normalization_190/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$rcnn_1/batch_normalization_190/gamma

8rcnn_1/batch_normalization_190/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_190/gamma*
_output_shapes	
:*
dtype0

#rcnn_1/batch_normalization_190/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#rcnn_1/batch_normalization_190/beta

7rcnn_1/batch_normalization_190/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_190/beta*
_output_shapes	
:*
dtype0
­
*rcnn_1/batch_normalization_190/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*rcnn_1/batch_normalization_190/moving_mean
¦
>rcnn_1/batch_normalization_190/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_190/moving_mean*
_output_shapes	
:*
dtype0
µ
.rcnn_1/batch_normalization_190/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.rcnn_1/batch_normalization_190/moving_variance
®
Brcnn_1/batch_normalization_190/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_190/moving_variance*
_output_shapes	
:*
dtype0

rcnn_1/conv2d_197/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*)
shared_namercnn_1/conv2d_197/kernel

,rcnn_1/conv2d_197/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_197/kernel*(
_output_shapes
:À*
dtype0

rcnn_1/conv2d_197/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*'
shared_namercnn_1/conv2d_197/bias
~
*rcnn_1/conv2d_197/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_197/bias*
_output_shapes	
:À*
dtype0
¡
$rcnn_1/batch_normalization_191/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*5
shared_name&$rcnn_1/batch_normalization_191/gamma

8rcnn_1/batch_normalization_191/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_191/gamma*
_output_shapes	
:À*
dtype0

#rcnn_1/batch_normalization_191/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*4
shared_name%#rcnn_1/batch_normalization_191/beta

7rcnn_1/batch_normalization_191/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_191/beta*
_output_shapes	
:À*
dtype0
­
*rcnn_1/batch_normalization_191/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*;
shared_name,*rcnn_1/batch_normalization_191/moving_mean
¦
>rcnn_1/batch_normalization_191/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_191/moving_mean*
_output_shapes	
:À*
dtype0
µ
.rcnn_1/batch_normalization_191/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*?
shared_name0.rcnn_1/batch_normalization_191/moving_variance
®
Brcnn_1/batch_normalization_191/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_191/moving_variance*
_output_shapes	
:À*
dtype0

rcnn_1/conv2d_198/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*)
shared_namercnn_1/conv2d_198/kernel

,rcnn_1/conv2d_198/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_198/kernel*(
_output_shapes
:À*
dtype0

rcnn_1/conv2d_198/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namercnn_1/conv2d_198/bias
~
*rcnn_1/conv2d_198/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_198/bias*
_output_shapes	
:*
dtype0
¡
$rcnn_1/batch_normalization_192/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$rcnn_1/batch_normalization_192/gamma

8rcnn_1/batch_normalization_192/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_192/gamma*
_output_shapes	
:*
dtype0

#rcnn_1/batch_normalization_192/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#rcnn_1/batch_normalization_192/beta

7rcnn_1/batch_normalization_192/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_192/beta*
_output_shapes	
:*
dtype0
­
*rcnn_1/batch_normalization_192/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*rcnn_1/batch_normalization_192/moving_mean
¦
>rcnn_1/batch_normalization_192/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_192/moving_mean*
_output_shapes	
:*
dtype0
µ
.rcnn_1/batch_normalization_192/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.rcnn_1/batch_normalization_192/moving_variance
®
Brcnn_1/batch_normalization_192/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_192/moving_variance*
_output_shapes	
:*
dtype0

rcnn_1/conv2d_199/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		0*)
shared_namercnn_1/conv2d_199/kernel

,rcnn_1/conv2d_199/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_199/kernel*'
_output_shapes
:		0*
dtype0

rcnn_1/conv2d_199/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_namercnn_1/conv2d_199/bias
}
*rcnn_1/conv2d_199/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_199/bias*
_output_shapes
:0*
dtype0
 
$rcnn_1/batch_normalization_193/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*5
shared_name&$rcnn_1/batch_normalization_193/gamma

8rcnn_1/batch_normalization_193/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_193/gamma*
_output_shapes
:0*
dtype0

#rcnn_1/batch_normalization_193/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#rcnn_1/batch_normalization_193/beta

7rcnn_1/batch_normalization_193/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_193/beta*
_output_shapes
:0*
dtype0
¬
*rcnn_1/batch_normalization_193/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*;
shared_name,*rcnn_1/batch_normalization_193/moving_mean
¥
>rcnn_1/batch_normalization_193/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_193/moving_mean*
_output_shapes
:0*
dtype0
´
.rcnn_1/batch_normalization_193/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.rcnn_1/batch_normalization_193/moving_variance
­
Brcnn_1/batch_normalization_193/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_193/moving_variance*
_output_shapes
:0*
dtype0

rcnn_1/conv2d_200/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*)
shared_namercnn_1/conv2d_200/kernel

,rcnn_1/conv2d_200/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_200/kernel*(
_output_shapes
:À*
dtype0

rcnn_1/conv2d_200/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namercnn_1/conv2d_200/bias
~
*rcnn_1/conv2d_200/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_200/bias*
_output_shapes	
:*
dtype0
¡
$rcnn_1/batch_normalization_194/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$rcnn_1/batch_normalization_194/gamma

8rcnn_1/batch_normalization_194/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_194/gamma*
_output_shapes	
:*
dtype0

#rcnn_1/batch_normalization_194/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#rcnn_1/batch_normalization_194/beta

7rcnn_1/batch_normalization_194/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_194/beta*
_output_shapes	
:*
dtype0
­
*rcnn_1/batch_normalization_194/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*rcnn_1/batch_normalization_194/moving_mean
¦
>rcnn_1/batch_normalization_194/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_194/moving_mean*
_output_shapes	
:*
dtype0
µ
.rcnn_1/batch_normalization_194/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.rcnn_1/batch_normalization_194/moving_variance
®
Brcnn_1/batch_normalization_194/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_194/moving_variance*
_output_shapes	
:*
dtype0

rcnn_1/conv2d_201/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_namercnn_1/conv2d_201/kernel

,rcnn_1/conv2d_201/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_201/kernel*'
_output_shapes
:0*
dtype0

rcnn_1/conv2d_201/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_namercnn_1/conv2d_201/bias
}
*rcnn_1/conv2d_201/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_201/bias*
_output_shapes
:0*
dtype0
 
$rcnn_1/batch_normalization_195/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*5
shared_name&$rcnn_1/batch_normalization_195/gamma

8rcnn_1/batch_normalization_195/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_195/gamma*
_output_shapes
:0*
dtype0

#rcnn_1/batch_normalization_195/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#rcnn_1/batch_normalization_195/beta

7rcnn_1/batch_normalization_195/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_195/beta*
_output_shapes
:0*
dtype0
¬
*rcnn_1/batch_normalization_195/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*;
shared_name,*rcnn_1/batch_normalization_195/moving_mean
¥
>rcnn_1/batch_normalization_195/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_195/moving_mean*
_output_shapes
:0*
dtype0
´
.rcnn_1/batch_normalization_195/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.rcnn_1/batch_normalization_195/moving_variance
­
Brcnn_1/batch_normalization_195/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_195/moving_variance*
_output_shapes
:0*
dtype0

rcnn_1/conv2d_202/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*)
shared_namercnn_1/conv2d_202/kernel

,rcnn_1/conv2d_202/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_202/kernel*(
_output_shapes
:À*
dtype0

rcnn_1/conv2d_202/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namercnn_1/conv2d_202/bias
~
*rcnn_1/conv2d_202/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_202/bias*
_output_shapes	
:*
dtype0
¡
$rcnn_1/batch_normalization_196/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$rcnn_1/batch_normalization_196/gamma

8rcnn_1/batch_normalization_196/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_196/gamma*
_output_shapes	
:*
dtype0

#rcnn_1/batch_normalization_196/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#rcnn_1/batch_normalization_196/beta

7rcnn_1/batch_normalization_196/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_196/beta*
_output_shapes	
:*
dtype0
­
*rcnn_1/batch_normalization_196/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*rcnn_1/batch_normalization_196/moving_mean
¦
>rcnn_1/batch_normalization_196/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_196/moving_mean*
_output_shapes	
:*
dtype0
µ
.rcnn_1/batch_normalization_196/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.rcnn_1/batch_normalization_196/moving_variance
®
Brcnn_1/batch_normalization_196/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_196/moving_variance*
_output_shapes	
:*
dtype0

rcnn_1/conv2d_203/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_namercnn_1/conv2d_203/kernel

,rcnn_1/conv2d_203/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_203/kernel*'
_output_shapes
:0*
dtype0

rcnn_1/conv2d_203/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_namercnn_1/conv2d_203/bias
}
*rcnn_1/conv2d_203/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_203/bias*
_output_shapes
:0*
dtype0
 
$rcnn_1/batch_normalization_197/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*5
shared_name&$rcnn_1/batch_normalization_197/gamma

8rcnn_1/batch_normalization_197/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_197/gamma*
_output_shapes
:0*
dtype0

#rcnn_1/batch_normalization_197/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#rcnn_1/batch_normalization_197/beta

7rcnn_1/batch_normalization_197/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_197/beta*
_output_shapes
:0*
dtype0
¬
*rcnn_1/batch_normalization_197/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*;
shared_name,*rcnn_1/batch_normalization_197/moving_mean
¥
>rcnn_1/batch_normalization_197/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_197/moving_mean*
_output_shapes
:0*
dtype0
´
.rcnn_1/batch_normalization_197/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.rcnn_1/batch_normalization_197/moving_variance
­
Brcnn_1/batch_normalization_197/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_197/moving_variance*
_output_shapes
:0*
dtype0

rcnn_1/conv2d_204/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*)
shared_namercnn_1/conv2d_204/kernel

,rcnn_1/conv2d_204/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_204/kernel*(
_output_shapes
:À*
dtype0

rcnn_1/conv2d_204/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namercnn_1/conv2d_204/bias
~
*rcnn_1/conv2d_204/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_204/bias*
_output_shapes	
:*
dtype0
¡
$rcnn_1/batch_normalization_198/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$rcnn_1/batch_normalization_198/gamma

8rcnn_1/batch_normalization_198/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_198/gamma*
_output_shapes	
:*
dtype0

#rcnn_1/batch_normalization_198/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#rcnn_1/batch_normalization_198/beta

7rcnn_1/batch_normalization_198/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_198/beta*
_output_shapes	
:*
dtype0
­
*rcnn_1/batch_normalization_198/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*rcnn_1/batch_normalization_198/moving_mean
¦
>rcnn_1/batch_normalization_198/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_198/moving_mean*
_output_shapes	
:*
dtype0
µ
.rcnn_1/batch_normalization_198/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.rcnn_1/batch_normalization_198/moving_variance
®
Brcnn_1/batch_normalization_198/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_198/moving_variance*
_output_shapes	
:*
dtype0

rcnn_1/conv2d_205/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_namercnn_1/conv2d_205/kernel

,rcnn_1/conv2d_205/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_205/kernel*'
_output_shapes
:0*
dtype0

rcnn_1/conv2d_205/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_namercnn_1/conv2d_205/bias
}
*rcnn_1/conv2d_205/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_205/bias*
_output_shapes
:0*
dtype0
 
$rcnn_1/batch_normalization_199/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*5
shared_name&$rcnn_1/batch_normalization_199/gamma

8rcnn_1/batch_normalization_199/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_199/gamma*
_output_shapes
:0*
dtype0

#rcnn_1/batch_normalization_199/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#rcnn_1/batch_normalization_199/beta

7rcnn_1/batch_normalization_199/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_199/beta*
_output_shapes
:0*
dtype0
¬
*rcnn_1/batch_normalization_199/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*;
shared_name,*rcnn_1/batch_normalization_199/moving_mean
¥
>rcnn_1/batch_normalization_199/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_199/moving_mean*
_output_shapes
:0*
dtype0
´
.rcnn_1/batch_normalization_199/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.rcnn_1/batch_normalization_199/moving_variance
­
Brcnn_1/batch_normalization_199/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_199/moving_variance*
_output_shapes
:0*
dtype0

rcnn_1/conv2d_206/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À*)
shared_namercnn_1/conv2d_206/kernel

,rcnn_1/conv2d_206/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_206/kernel*(
_output_shapes
:À*
dtype0

rcnn_1/conv2d_206/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namercnn_1/conv2d_206/bias
~
*rcnn_1/conv2d_206/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_206/bias*
_output_shapes	
:*
dtype0
¡
$rcnn_1/batch_normalization_200/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$rcnn_1/batch_normalization_200/gamma

8rcnn_1/batch_normalization_200/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_200/gamma*
_output_shapes	
:*
dtype0

#rcnn_1/batch_normalization_200/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#rcnn_1/batch_normalization_200/beta

7rcnn_1/batch_normalization_200/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_200/beta*
_output_shapes	
:*
dtype0
­
*rcnn_1/batch_normalization_200/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*rcnn_1/batch_normalization_200/moving_mean
¦
>rcnn_1/batch_normalization_200/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_200/moving_mean*
_output_shapes	
:*
dtype0
µ
.rcnn_1/batch_normalization_200/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.rcnn_1/batch_normalization_200/moving_variance
®
Brcnn_1/batch_normalization_200/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_200/moving_variance*
_output_shapes	
:*
dtype0

rcnn_1/conv2d_207/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_namercnn_1/conv2d_207/kernel

,rcnn_1/conv2d_207/kernel/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_207/kernel*'
_output_shapes
:0*
dtype0

rcnn_1/conv2d_207/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_namercnn_1/conv2d_207/bias
}
*rcnn_1/conv2d_207/bias/Read/ReadVariableOpReadVariableOprcnn_1/conv2d_207/bias*
_output_shapes
:0*
dtype0
 
$rcnn_1/batch_normalization_201/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*5
shared_name&$rcnn_1/batch_normalization_201/gamma

8rcnn_1/batch_normalization_201/gamma/Read/ReadVariableOpReadVariableOp$rcnn_1/batch_normalization_201/gamma*
_output_shapes
:0*
dtype0

#rcnn_1/batch_normalization_201/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*4
shared_name%#rcnn_1/batch_normalization_201/beta

7rcnn_1/batch_normalization_201/beta/Read/ReadVariableOpReadVariableOp#rcnn_1/batch_normalization_201/beta*
_output_shapes
:0*
dtype0
¬
*rcnn_1/batch_normalization_201/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*;
shared_name,*rcnn_1/batch_normalization_201/moving_mean
¥
>rcnn_1/batch_normalization_201/moving_mean/Read/ReadVariableOpReadVariableOp*rcnn_1/batch_normalization_201/moving_mean*
_output_shapes
:0*
dtype0
´
.rcnn_1/batch_normalization_201/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.rcnn_1/batch_normalization_201/moving_variance
­
Brcnn_1/batch_normalization_201/moving_variance/Read/ReadVariableOpReadVariableOp.rcnn_1/batch_normalization_201/moving_variance*
_output_shapes
:0*
dtype0

rcnn_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À*&
shared_namercnn_1/dense_6/kernel

)rcnn_1/dense_6/kernel/Read/ReadVariableOpReadVariableOprcnn_1/dense_6/kernel* 
_output_shapes
:
À*
dtype0

rcnn_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namercnn_1/dense_6/bias
x
'rcnn_1/dense_6/bias/Read/ReadVariableOpReadVariableOprcnn_1/dense_6/bias*
_output_shapes	
:*
dtype0

rcnn_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_namercnn_1/dense_7/kernel

)rcnn_1/dense_7/kernel/Read/ReadVariableOpReadVariableOprcnn_1/dense_7/kernel*
_output_shapes
:	@*
dtype0
~
rcnn_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namercnn_1/dense_7/bias
w
'rcnn_1/dense_7/bias/Read/ReadVariableOpReadVariableOprcnn_1/dense_7/bias*
_output_shapes
:@*
dtype0

rcnn_1/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_namercnn_1/dense_8/kernel

)rcnn_1/dense_8/kernel/Read/ReadVariableOpReadVariableOprcnn_1/dense_8/kernel*
_output_shapes

:@*
dtype0
~
rcnn_1/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namercnn_1/dense_8/bias
w
'rcnn_1/dense_8/bias/Read/ReadVariableOpReadVariableOprcnn_1/dense_8/bias*
_output_shapes
:*
dtype0

rcnn_1/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À*&
shared_namercnn_1/dense_9/kernel

)rcnn_1/dense_9/kernel/Read/ReadVariableOpReadVariableOprcnn_1/dense_9/kernel* 
_output_shapes
:
À*
dtype0

rcnn_1/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namercnn_1/dense_9/bias
x
'rcnn_1/dense_9/bias/Read/ReadVariableOpReadVariableOprcnn_1/dense_9/bias*
_output_shapes	
:*
dtype0

rcnn_1/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_namercnn_1/dense_10/kernel

*rcnn_1/dense_10/kernel/Read/ReadVariableOpReadVariableOprcnn_1/dense_10/kernel* 
_output_shapes
:
*
dtype0

rcnn_1/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namercnn_1/dense_10/bias
z
(rcnn_1/dense_10/bias/Read/ReadVariableOpReadVariableOprcnn_1/dense_10/bias*
_output_shapes	
:*
dtype0

rcnn_1/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_namercnn_1/dense_11/kernel

*rcnn_1/dense_11/kernel/Read/ReadVariableOpReadVariableOprcnn_1/dense_11/kernel*
_output_shapes
:	*
dtype0

rcnn_1/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namercnn_1/dense_11/bias
y
(rcnn_1/dense_11/bias/Read/ReadVariableOpReadVariableOprcnn_1/dense_11/bias*
_output_shapes
:*
dtype0
¶
ConstConst*
_output_shapes
:*
dtype0	*}
valuetBr	"h                                                                                                   
¸
Const_1Const*
_output_shapes
:*
dtype0	*}
valuetBr	"h                                                                                                   

NoOpNoOp
ÏÀ
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*À
valueü¿Bø¿ Bð¿

conv2d_condense1
batch_norm1
conv2d_condense2
batch_norm2
conv2d_extract12_condense
batch_norm_extract12a
conv2d_extract12
batch_norm_extract12b
	maxpool_extract17_3

conv2d_extract8_condense
batch_norm_extract8a
conv2d_extract8
batch_norm_extract8b
maxpool_extract12_3
conv2d_extract5_condense
batch_norm_extract5a
conv2d_extract5
batch_norm_extract5b
maxpool_extract8_3
conv2d_extract3_condense
batch_norm_extract3a
conv2d_extract3
batch_norm_extract3b
maxpool_extract5_3
conv2d_extract2_condense
batch_norm_extract2a
conv2d_extract2
batch_norm_extract2b
flatten
classifier1
classifier2
 classifier3
!
regressor1
"
regressor2
#
regressor3
$	variables
%regularization_losses
&trainable_variables
'	keras_api
(
signatures
h

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api

/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api

>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
h

Gkernel
Hbias
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api

Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api

\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
R
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
h

ikernel
jbias
k	variables
lregularization_losses
mtrainable_variables
n	keras_api

oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
h

xkernel
ybias
z	variables
{regularization_losses
|trainable_variables
}	keras_api

~axis
	gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
 
	 axis

¡gamma
	¢beta
£moving_mean
¤moving_variance
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
V
©	variables
ªregularization_losses
«trainable_variables
¬	keras_api
n
­kernel
	®bias
¯	variables
°regularization_losses
±trainable_variables
²	keras_api
 
	³axis

´gamma
	µbeta
¶moving_mean
·moving_variance
¸	variables
¹regularization_losses
ºtrainable_variables
»	keras_api
n
¼kernel
	½bias
¾	variables
¿regularization_losses
Àtrainable_variables
Á	keras_api
 
	Âaxis

Ãgamma
	Äbeta
Åmoving_mean
Æmoving_variance
Ç	variables
Èregularization_losses
Étrainable_variables
Ê	keras_api
V
Ë	variables
Ìregularization_losses
Ítrainable_variables
Î	keras_api
n
Ïkernel
	Ðbias
Ñ	variables
Òregularization_losses
Ótrainable_variables
Ô	keras_api
 
	Õaxis

Ögamma
	×beta
Ømoving_mean
Ùmoving_variance
Ú	variables
Ûregularization_losses
Ütrainable_variables
Ý	keras_api
n
Þkernel
	ßbias
à	variables
áregularization_losses
âtrainable_variables
ã	keras_api
 
	äaxis

ågamma
	æbeta
çmoving_mean
èmoving_variance
é	variables
êregularization_losses
ëtrainable_variables
ì	keras_api
V
í	variables
îregularization_losses
ïtrainable_variables
ð	keras_api
n
ñkernel
	òbias
ó	variables
ôregularization_losses
õtrainable_variables
ö	keras_api
n
÷kernel
	øbias
ù	variables
úregularization_losses
ûtrainable_variables
ü	keras_api
n
ýkernel
	þbias
ÿ	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
É
)0
*1
02
13
24
35
86
97
?8
@9
A10
B11
G12
H13
N14
O15
P16
Q17
V18
W19
]20
^21
_22
`23
i24
j25
p26
q27
r28
s29
x30
y31
32
33
34
35
36
37
38
39
40
41
42
43
¡44
¢45
£46
¤47
­48
®49
´50
µ51
¶52
·53
¼54
½55
Ã56
Ä57
Å58
Æ59
Ï60
Ð61
Ö62
×63
Ø64
Ù65
Þ66
ß67
å68
æ69
ç70
è71
ñ72
ò73
÷74
ø75
ý76
þ77
78
79
80
81
82
83
 
û
)0
*1
02
13
84
95
?6
@7
G8
H9
N10
O11
V12
W13
]14
^15
i16
j17
p18
q19
x20
y21
22
23
24
25
26
27
28
29
¡30
¢31
­32
®33
´34
µ35
¼36
½37
Ã38
Ä39
Ï40
Ð41
Ö42
×43
Þ44
ß45
å46
æ47
ñ48
ò49
÷50
ø51
ý52
þ53
54
55
56
57
58
59
²
non_trainable_variables
$	variables
layers
%regularization_losses
 layer_regularization_losses
layer_metrics
&trainable_variables
metrics
 
`^
VARIABLE_VALUErcnn_1/conv2d_196/kernel2conv2d_condense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUErcnn_1/conv2d_196/bias0conv2d_condense1/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
²
non_trainable_variables
+	variables
layers
,regularization_losses
 layer_regularization_losses
layer_metrics
-trainable_variables
metrics
 
fd
VARIABLE_VALUE$rcnn_1/batch_normalization_190/gamma,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE#rcnn_1/batch_normalization_190/beta+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE*rcnn_1/batch_normalization_190/moving_mean2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE.rcnn_1/batch_normalization_190/moving_variance6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
22
33
 

00
11
²
non_trainable_variables
4	variables
 layers
5regularization_losses
 ¡layer_regularization_losses
¢layer_metrics
6trainable_variables
£metrics
`^
VARIABLE_VALUErcnn_1/conv2d_197/kernel2conv2d_condense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUErcnn_1/conv2d_197/bias0conv2d_condense2/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
²
¤non_trainable_variables
:	variables
¥layers
;regularization_losses
 ¦layer_regularization_losses
§layer_metrics
<trainable_variables
¨metrics
 
fd
VARIABLE_VALUE$rcnn_1/batch_normalization_191/gamma,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE#rcnn_1/batch_normalization_191/beta+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE*rcnn_1/batch_normalization_191/moving_mean2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE.rcnn_1/batch_normalization_191/moving_variance6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
A2
B3
 

?0
@1
²
©non_trainable_variables
C	variables
ªlayers
Dregularization_losses
 «layer_regularization_losses
¬layer_metrics
Etrainable_variables
­metrics
ig
VARIABLE_VALUErcnn_1/conv2d_198/kernel;conv2d_extract12_condense/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUErcnn_1/conv2d_198/bias9conv2d_extract12_condense/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
²
®non_trainable_variables
I	variables
¯layers
Jregularization_losses
 °layer_regularization_losses
±layer_metrics
Ktrainable_variables
²metrics
 
pn
VARIABLE_VALUE$rcnn_1/batch_normalization_192/gamma6batch_norm_extract12a/gamma/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE#rcnn_1/batch_normalization_192/beta5batch_norm_extract12a/beta/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE*rcnn_1/batch_normalization_192/moving_mean<batch_norm_extract12a/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_192/moving_variance@batch_norm_extract12a/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
P2
Q3
 

N0
O1
²
³non_trainable_variables
R	variables
´layers
Sregularization_losses
 µlayer_regularization_losses
¶layer_metrics
Ttrainable_variables
·metrics
`^
VARIABLE_VALUErcnn_1/conv2d_199/kernel2conv2d_extract12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUErcnn_1/conv2d_199/bias0conv2d_extract12/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
²
¸non_trainable_variables
X	variables
¹layers
Yregularization_losses
 ºlayer_regularization_losses
»layer_metrics
Ztrainable_variables
¼metrics
 
pn
VARIABLE_VALUE$rcnn_1/batch_normalization_193/gamma6batch_norm_extract12b/gamma/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE#rcnn_1/batch_normalization_193/beta5batch_norm_extract12b/beta/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE*rcnn_1/batch_normalization_193/moving_mean<batch_norm_extract12b/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_193/moving_variance@batch_norm_extract12b/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
_2
`3
 

]0
^1
²
½non_trainable_variables
a	variables
¾layers
bregularization_losses
 ¿layer_regularization_losses
Àlayer_metrics
ctrainable_variables
Ámetrics
 
 
 
²
Ânon_trainable_variables
e	variables
Ãlayers
fregularization_losses
 Älayer_regularization_losses
Ålayer_metrics
gtrainable_variables
Æmetrics
hf
VARIABLE_VALUErcnn_1/conv2d_200/kernel:conv2d_extract8_condense/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUErcnn_1/conv2d_200/bias8conv2d_extract8_condense/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
²
Çnon_trainable_variables
k	variables
Èlayers
lregularization_losses
 Élayer_regularization_losses
Êlayer_metrics
mtrainable_variables
Ëmetrics
 
om
VARIABLE_VALUE$rcnn_1/batch_normalization_194/gamma5batch_norm_extract8a/gamma/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#rcnn_1/batch_normalization_194/beta4batch_norm_extract8a/beta/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE*rcnn_1/batch_normalization_194/moving_mean;batch_norm_extract8a/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_194/moving_variance?batch_norm_extract8a/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
r2
s3
 

p0
q1
²
Ìnon_trainable_variables
t	variables
Ílayers
uregularization_losses
 Îlayer_regularization_losses
Ïlayer_metrics
vtrainable_variables
Ðmetrics
_]
VARIABLE_VALUErcnn_1/conv2d_201/kernel1conv2d_extract8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErcnn_1/conv2d_201/bias/conv2d_extract8/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
 

x0
y1
²
Ñnon_trainable_variables
z	variables
Òlayers
{regularization_losses
 Ólayer_regularization_losses
Ôlayer_metrics
|trainable_variables
Õmetrics
 
om
VARIABLE_VALUE$rcnn_1/batch_normalization_195/gamma5batch_norm_extract8b/gamma/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#rcnn_1/batch_normalization_195/beta4batch_norm_extract8b/beta/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE*rcnn_1/batch_normalization_195/moving_mean;batch_norm_extract8b/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_195/moving_variance?batch_norm_extract8b/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1
µ
Önon_trainable_variables
	variables
×layers
regularization_losses
 Ølayer_regularization_losses
Ùlayer_metrics
trainable_variables
Úmetrics
 
 
 
µ
Ûnon_trainable_variables
	variables
Ülayers
regularization_losses
 Ýlayer_regularization_losses
Þlayer_metrics
trainable_variables
ßmetrics
hf
VARIABLE_VALUErcnn_1/conv2d_202/kernel:conv2d_extract5_condense/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUErcnn_1/conv2d_202/bias8conv2d_extract5_condense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
ànon_trainable_variables
	variables
álayers
regularization_losses
 âlayer_regularization_losses
ãlayer_metrics
trainable_variables
ämetrics
 
om
VARIABLE_VALUE$rcnn_1/batch_normalization_196/gamma5batch_norm_extract5a/gamma/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#rcnn_1/batch_normalization_196/beta4batch_norm_extract5a/beta/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE*rcnn_1/batch_normalization_196/moving_mean;batch_norm_extract5a/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_196/moving_variance?batch_norm_extract5a/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3
 

0
1
µ
ånon_trainable_variables
	variables
ælayers
regularization_losses
 çlayer_regularization_losses
èlayer_metrics
trainable_variables
émetrics
_]
VARIABLE_VALUErcnn_1/conv2d_203/kernel1conv2d_extract5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErcnn_1/conv2d_203/bias/conv2d_extract5/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
ênon_trainable_variables
	variables
ëlayers
regularization_losses
 ìlayer_regularization_losses
ílayer_metrics
trainable_variables
îmetrics
 
om
VARIABLE_VALUE$rcnn_1/batch_normalization_197/gamma5batch_norm_extract5b/gamma/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#rcnn_1/batch_normalization_197/beta4batch_norm_extract5b/beta/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE*rcnn_1/batch_normalization_197/moving_mean;batch_norm_extract5b/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_197/moving_variance?batch_norm_extract5b/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
¡0
¢1
£2
¤3
 

¡0
¢1
µ
ïnon_trainable_variables
¥	variables
ðlayers
¦regularization_losses
 ñlayer_regularization_losses
òlayer_metrics
§trainable_variables
ómetrics
 
 
 
µ
ônon_trainable_variables
©	variables
õlayers
ªregularization_losses
 ölayer_regularization_losses
÷layer_metrics
«trainable_variables
ømetrics
hf
VARIABLE_VALUErcnn_1/conv2d_204/kernel:conv2d_extract3_condense/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUErcnn_1/conv2d_204/bias8conv2d_extract3_condense/bias/.ATTRIBUTES/VARIABLE_VALUE

­0
®1
 

­0
®1
µ
ùnon_trainable_variables
¯	variables
úlayers
°regularization_losses
 ûlayer_regularization_losses
ülayer_metrics
±trainable_variables
ýmetrics
 
om
VARIABLE_VALUE$rcnn_1/batch_normalization_198/gamma5batch_norm_extract3a/gamma/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#rcnn_1/batch_normalization_198/beta4batch_norm_extract3a/beta/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE*rcnn_1/batch_normalization_198/moving_mean;batch_norm_extract3a/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_198/moving_variance?batch_norm_extract3a/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
´0
µ1
¶2
·3
 

´0
µ1
µ
þnon_trainable_variables
¸	variables
ÿlayers
¹regularization_losses
 layer_regularization_losses
layer_metrics
ºtrainable_variables
metrics
_]
VARIABLE_VALUErcnn_1/conv2d_205/kernel1conv2d_extract3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErcnn_1/conv2d_205/bias/conv2d_extract3/bias/.ATTRIBUTES/VARIABLE_VALUE

¼0
½1
 

¼0
½1
µ
non_trainable_variables
¾	variables
layers
¿regularization_losses
 layer_regularization_losses
layer_metrics
Àtrainable_variables
metrics
 
om
VARIABLE_VALUE$rcnn_1/batch_normalization_199/gamma5batch_norm_extract3b/gamma/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#rcnn_1/batch_normalization_199/beta4batch_norm_extract3b/beta/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE*rcnn_1/batch_normalization_199/moving_mean;batch_norm_extract3b/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_199/moving_variance?batch_norm_extract3b/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ã0
Ä1
Å2
Æ3
 

Ã0
Ä1
µ
non_trainable_variables
Ç	variables
layers
Èregularization_losses
 layer_regularization_losses
layer_metrics
Étrainable_variables
metrics
 
 
 
µ
non_trainable_variables
Ë	variables
layers
Ìregularization_losses
 layer_regularization_losses
layer_metrics
Ítrainable_variables
metrics
hf
VARIABLE_VALUErcnn_1/conv2d_206/kernel:conv2d_extract2_condense/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUErcnn_1/conv2d_206/bias8conv2d_extract2_condense/bias/.ATTRIBUTES/VARIABLE_VALUE

Ï0
Ð1
 

Ï0
Ð1
µ
non_trainable_variables
Ñ	variables
layers
Òregularization_losses
 layer_regularization_losses
layer_metrics
Ótrainable_variables
metrics
 
om
VARIABLE_VALUE$rcnn_1/batch_normalization_200/gamma5batch_norm_extract2a/gamma/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#rcnn_1/batch_normalization_200/beta4batch_norm_extract2a/beta/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE*rcnn_1/batch_normalization_200/moving_mean;batch_norm_extract2a/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_200/moving_variance?batch_norm_extract2a/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ö0
×1
Ø2
Ù3
 

Ö0
×1
µ
non_trainable_variables
Ú	variables
layers
Ûregularization_losses
 layer_regularization_losses
layer_metrics
Ütrainable_variables
metrics
_]
VARIABLE_VALUErcnn_1/conv2d_207/kernel1conv2d_extract2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErcnn_1/conv2d_207/bias/conv2d_extract2/bias/.ATTRIBUTES/VARIABLE_VALUE

Þ0
ß1
 

Þ0
ß1
µ
non_trainable_variables
à	variables
layers
áregularization_losses
 layer_regularization_losses
layer_metrics
âtrainable_variables
 metrics
 
om
VARIABLE_VALUE$rcnn_1/batch_normalization_201/gamma5batch_norm_extract2b/gamma/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE#rcnn_1/batch_normalization_201/beta4batch_norm_extract2b/beta/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE*rcnn_1/batch_normalization_201/moving_mean;batch_norm_extract2b/moving_mean/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.rcnn_1/batch_normalization_201/moving_variance?batch_norm_extract2b/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
å0
æ1
ç2
è3
 

å0
æ1
µ
¡non_trainable_variables
é	variables
¢layers
êregularization_losses
 £layer_regularization_losses
¤layer_metrics
ëtrainable_variables
¥metrics
 
 
 
µ
¦non_trainable_variables
í	variables
§layers
îregularization_losses
 ¨layer_regularization_losses
©layer_metrics
ïtrainable_variables
ªmetrics
XV
VARIABLE_VALUErcnn_1/dense_6/kernel-classifier1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUErcnn_1/dense_6/bias+classifier1/bias/.ATTRIBUTES/VARIABLE_VALUE

ñ0
ò1
 

ñ0
ò1
µ
«non_trainable_variables
ó	variables
¬layers
ôregularization_losses
 ­layer_regularization_losses
®layer_metrics
õtrainable_variables
¯metrics
XV
VARIABLE_VALUErcnn_1/dense_7/kernel-classifier2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUErcnn_1/dense_7/bias+classifier2/bias/.ATTRIBUTES/VARIABLE_VALUE

÷0
ø1
 

÷0
ø1
µ
°non_trainable_variables
ù	variables
±layers
úregularization_losses
 ²layer_regularization_losses
³layer_metrics
ûtrainable_variables
´metrics
XV
VARIABLE_VALUErcnn_1/dense_8/kernel-classifier3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUErcnn_1/dense_8/bias+classifier3/bias/.ATTRIBUTES/VARIABLE_VALUE

ý0
þ1
 

ý0
þ1
µ
µnon_trainable_variables
ÿ	variables
¶layers
regularization_losses
 ·layer_regularization_losses
¸layer_metrics
trainable_variables
¹metrics
WU
VARIABLE_VALUErcnn_1/dense_9/kernel,regressor1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUErcnn_1/dense_9/bias*regressor1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
ºnon_trainable_variables
	variables
»layers
regularization_losses
 ¼layer_regularization_losses
½layer_metrics
trainable_variables
¾metrics
XV
VARIABLE_VALUErcnn_1/dense_10/kernel,regressor2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUErcnn_1/dense_10/bias*regressor2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
¿non_trainable_variables
	variables
Àlayers
regularization_losses
 Álayer_regularization_losses
Âlayer_metrics
trainable_variables
Ãmetrics
XV
VARIABLE_VALUErcnn_1/dense_11/kernel,regressor3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUErcnn_1/dense_11/bias*regressor3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
Änon_trainable_variables
	variables
Ålayers
regularization_losses
 Ælayer_regularization_losses
Çlayer_metrics
trainable_variables
Èmetrics
Ä
20
31
A2
B3
P4
Q5
_6
`7
r8
s9
10
11
12
13
£14
¤15
¶16
·17
Å18
Æ19
Ø20
Ù21
ç22
è23

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 

P0
Q1
 
 
 
 
 
 
 
 
 

_0
`1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

r0
s1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

£0
¤1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

¶0
·1
 
 
 
 
 
 
 
 
 

Å0
Æ1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Ø0
Ù1
 
 
 
 
 
 
 
 
 

ç0
è1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_input_2_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_input_2_2Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_input_2_3Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_input_2_4Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_input_2_5Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
Ê
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2_1serving_default_input_2_2serving_default_input_2_3serving_default_input_2_4serving_default_input_2_5rcnn_1/conv2d_196/kernelrcnn_1/conv2d_196/bias$rcnn_1/batch_normalization_190/gamma#rcnn_1/batch_normalization_190/beta*rcnn_1/batch_normalization_190/moving_mean.rcnn_1/batch_normalization_190/moving_variancercnn_1/conv2d_197/kernelrcnn_1/conv2d_197/bias$rcnn_1/batch_normalization_191/gamma#rcnn_1/batch_normalization_191/beta*rcnn_1/batch_normalization_191/moving_mean.rcnn_1/batch_normalization_191/moving_varianceConstConst_1rcnn_1/conv2d_206/kernelrcnn_1/conv2d_206/bias$rcnn_1/batch_normalization_200/gamma#rcnn_1/batch_normalization_200/beta*rcnn_1/batch_normalization_200/moving_mean.rcnn_1/batch_normalization_200/moving_variancercnn_1/conv2d_207/kernelrcnn_1/conv2d_207/bias$rcnn_1/batch_normalization_201/gamma#rcnn_1/batch_normalization_201/beta*rcnn_1/batch_normalization_201/moving_mean.rcnn_1/batch_normalization_201/moving_variancercnn_1/conv2d_204/kernelrcnn_1/conv2d_204/bias$rcnn_1/batch_normalization_198/gamma#rcnn_1/batch_normalization_198/beta*rcnn_1/batch_normalization_198/moving_mean.rcnn_1/batch_normalization_198/moving_variancercnn_1/conv2d_205/kernelrcnn_1/conv2d_205/bias$rcnn_1/batch_normalization_199/gamma#rcnn_1/batch_normalization_199/beta*rcnn_1/batch_normalization_199/moving_mean.rcnn_1/batch_normalization_199/moving_variancercnn_1/conv2d_202/kernelrcnn_1/conv2d_202/bias$rcnn_1/batch_normalization_196/gamma#rcnn_1/batch_normalization_196/beta*rcnn_1/batch_normalization_196/moving_mean.rcnn_1/batch_normalization_196/moving_variancercnn_1/conv2d_203/kernelrcnn_1/conv2d_203/bias$rcnn_1/batch_normalization_197/gamma#rcnn_1/batch_normalization_197/beta*rcnn_1/batch_normalization_197/moving_mean.rcnn_1/batch_normalization_197/moving_variancercnn_1/conv2d_200/kernelrcnn_1/conv2d_200/bias$rcnn_1/batch_normalization_194/gamma#rcnn_1/batch_normalization_194/beta*rcnn_1/batch_normalization_194/moving_mean.rcnn_1/batch_normalization_194/moving_variancercnn_1/conv2d_201/kernelrcnn_1/conv2d_201/bias$rcnn_1/batch_normalization_195/gamma#rcnn_1/batch_normalization_195/beta*rcnn_1/batch_normalization_195/moving_mean.rcnn_1/batch_normalization_195/moving_variancercnn_1/conv2d_198/kernelrcnn_1/conv2d_198/bias$rcnn_1/batch_normalization_192/gamma#rcnn_1/batch_normalization_192/beta*rcnn_1/batch_normalization_192/moving_mean.rcnn_1/batch_normalization_192/moving_variancercnn_1/conv2d_199/kernelrcnn_1/conv2d_199/bias$rcnn_1/batch_normalization_193/gamma#rcnn_1/batch_normalization_193/beta*rcnn_1/batch_normalization_193/moving_mean.rcnn_1/batch_normalization_193/moving_variancercnn_1/dense_6/kernelrcnn_1/dense_6/biasrcnn_1/dense_7/kernelrcnn_1/dense_7/biasrcnn_1/dense_8/kernelrcnn_1/dense_8/biasrcnn_1/dense_9/kernelrcnn_1/dense_9/biasrcnn_1/dense_10/kernelrcnn_1/dense_10/biasrcnn_1/dense_11/kernelrcnn_1/dense_11/bias*g
Tin`
^2\		*
Tout
2					*
_collective_manager_ids
 *³
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*v
_read_only_resource_inputsX
VT	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_59613
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¸&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,rcnn_1/conv2d_196/kernel/Read/ReadVariableOp*rcnn_1/conv2d_196/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_190/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_190/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_190/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_190/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_197/kernel/Read/ReadVariableOp*rcnn_1/conv2d_197/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_191/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_191/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_191/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_191/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_198/kernel/Read/ReadVariableOp*rcnn_1/conv2d_198/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_192/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_192/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_192/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_192/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_199/kernel/Read/ReadVariableOp*rcnn_1/conv2d_199/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_193/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_193/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_193/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_193/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_200/kernel/Read/ReadVariableOp*rcnn_1/conv2d_200/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_194/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_194/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_194/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_194/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_201/kernel/Read/ReadVariableOp*rcnn_1/conv2d_201/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_195/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_195/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_195/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_195/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_202/kernel/Read/ReadVariableOp*rcnn_1/conv2d_202/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_196/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_196/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_196/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_196/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_203/kernel/Read/ReadVariableOp*rcnn_1/conv2d_203/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_197/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_197/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_197/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_197/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_204/kernel/Read/ReadVariableOp*rcnn_1/conv2d_204/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_198/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_198/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_198/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_198/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_205/kernel/Read/ReadVariableOp*rcnn_1/conv2d_205/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_199/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_199/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_199/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_199/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_206/kernel/Read/ReadVariableOp*rcnn_1/conv2d_206/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_200/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_200/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_200/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_200/moving_variance/Read/ReadVariableOp,rcnn_1/conv2d_207/kernel/Read/ReadVariableOp*rcnn_1/conv2d_207/bias/Read/ReadVariableOp8rcnn_1/batch_normalization_201/gamma/Read/ReadVariableOp7rcnn_1/batch_normalization_201/beta/Read/ReadVariableOp>rcnn_1/batch_normalization_201/moving_mean/Read/ReadVariableOpBrcnn_1/batch_normalization_201/moving_variance/Read/ReadVariableOp)rcnn_1/dense_6/kernel/Read/ReadVariableOp'rcnn_1/dense_6/bias/Read/ReadVariableOp)rcnn_1/dense_7/kernel/Read/ReadVariableOp'rcnn_1/dense_7/bias/Read/ReadVariableOp)rcnn_1/dense_8/kernel/Read/ReadVariableOp'rcnn_1/dense_8/bias/Read/ReadVariableOp)rcnn_1/dense_9/kernel/Read/ReadVariableOp'rcnn_1/dense_9/bias/Read/ReadVariableOp*rcnn_1/dense_10/kernel/Read/ReadVariableOp(rcnn_1/dense_10/bias/Read/ReadVariableOp*rcnn_1/dense_11/kernel/Read/ReadVariableOp(rcnn_1/dense_11/bias/Read/ReadVariableOpConst_2*a
TinZ
X2V*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_65971
¡
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamercnn_1/conv2d_196/kernelrcnn_1/conv2d_196/bias$rcnn_1/batch_normalization_190/gamma#rcnn_1/batch_normalization_190/beta*rcnn_1/batch_normalization_190/moving_mean.rcnn_1/batch_normalization_190/moving_variancercnn_1/conv2d_197/kernelrcnn_1/conv2d_197/bias$rcnn_1/batch_normalization_191/gamma#rcnn_1/batch_normalization_191/beta*rcnn_1/batch_normalization_191/moving_mean.rcnn_1/batch_normalization_191/moving_variancercnn_1/conv2d_198/kernelrcnn_1/conv2d_198/bias$rcnn_1/batch_normalization_192/gamma#rcnn_1/batch_normalization_192/beta*rcnn_1/batch_normalization_192/moving_mean.rcnn_1/batch_normalization_192/moving_variancercnn_1/conv2d_199/kernelrcnn_1/conv2d_199/bias$rcnn_1/batch_normalization_193/gamma#rcnn_1/batch_normalization_193/beta*rcnn_1/batch_normalization_193/moving_mean.rcnn_1/batch_normalization_193/moving_variancercnn_1/conv2d_200/kernelrcnn_1/conv2d_200/bias$rcnn_1/batch_normalization_194/gamma#rcnn_1/batch_normalization_194/beta*rcnn_1/batch_normalization_194/moving_mean.rcnn_1/batch_normalization_194/moving_variancercnn_1/conv2d_201/kernelrcnn_1/conv2d_201/bias$rcnn_1/batch_normalization_195/gamma#rcnn_1/batch_normalization_195/beta*rcnn_1/batch_normalization_195/moving_mean.rcnn_1/batch_normalization_195/moving_variancercnn_1/conv2d_202/kernelrcnn_1/conv2d_202/bias$rcnn_1/batch_normalization_196/gamma#rcnn_1/batch_normalization_196/beta*rcnn_1/batch_normalization_196/moving_mean.rcnn_1/batch_normalization_196/moving_variancercnn_1/conv2d_203/kernelrcnn_1/conv2d_203/bias$rcnn_1/batch_normalization_197/gamma#rcnn_1/batch_normalization_197/beta*rcnn_1/batch_normalization_197/moving_mean.rcnn_1/batch_normalization_197/moving_variancercnn_1/conv2d_204/kernelrcnn_1/conv2d_204/bias$rcnn_1/batch_normalization_198/gamma#rcnn_1/batch_normalization_198/beta*rcnn_1/batch_normalization_198/moving_mean.rcnn_1/batch_normalization_198/moving_variancercnn_1/conv2d_205/kernelrcnn_1/conv2d_205/bias$rcnn_1/batch_normalization_199/gamma#rcnn_1/batch_normalization_199/beta*rcnn_1/batch_normalization_199/moving_mean.rcnn_1/batch_normalization_199/moving_variancercnn_1/conv2d_206/kernelrcnn_1/conv2d_206/bias$rcnn_1/batch_normalization_200/gamma#rcnn_1/batch_normalization_200/beta*rcnn_1/batch_normalization_200/moving_mean.rcnn_1/batch_normalization_200/moving_variancercnn_1/conv2d_207/kernelrcnn_1/conv2d_207/bias$rcnn_1/batch_normalization_201/gamma#rcnn_1/batch_normalization_201/beta*rcnn_1/batch_normalization_201/moving_mean.rcnn_1/batch_normalization_201/moving_variancercnn_1/dense_6/kernelrcnn_1/dense_6/biasrcnn_1/dense_7/kernelrcnn_1/dense_7/biasrcnn_1/dense_8/kernelrcnn_1/dense_8/biasrcnn_1/dense_9/kernelrcnn_1/dense_9/biasrcnn_1/dense_10/kernelrcnn_1/dense_10/biasrcnn_1/dense_11/kernelrcnn_1/dense_11/bias*`
TinY
W2U*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_66233½ÿ


*__inference_conv2d_201_layer_call_fn_64529

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_201_layer_call_and_return_conditional_losses_563302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64271

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Û
ª
7__inference_batch_normalization_198_layer_call_fn_65088

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_557172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_199_layer_call_fn_64233

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_199_layer_call_and_return_conditional_losses_566042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_54747

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_56702

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_54778

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_53879

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ª
7__inference_batch_normalization_193_layer_call_fn_64284

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_541912
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý
ª
7__inference_batch_normalization_190_layer_call_fn_63917

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_551862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64613

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
	
­
E__inference_conv2d_205_layer_call_and_return_conditional_losses_55782

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65455

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ù
ª
7__inference_batch_normalization_195_layer_call_fn_64593

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_563832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
	
­
E__inference_conv2d_206_layer_call_and_return_conditional_losses_65260

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ý
ª
7__inference_batch_normalization_192_layer_call_fn_64149

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_565572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
ª
7__inference_batch_normalization_199_layer_call_fn_65249

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_558352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65437

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ø
|
'__inference_dense_8_layer_call_fn_65616

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_568032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
­
L
0__inference_max_pooling2d_11_layer_call_fn_54905

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_548992
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_64039

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_55286

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_54459

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_6_layer_call_and_return_conditional_losses_65567

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Û
ª
7__inference_batch_normalization_190_layer_call_fn_63904

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_551682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
ª
7__inference_batch_normalization_190_layer_call_fn_63853

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_539102
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_204_layer_call_fn_64973

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_204_layer_call_and_return_conditional_losses_556822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_55817

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs


*__inference_conv2d_202_layer_call_fn_64677

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_202_layer_call_and_return_conditional_losses_559562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64927

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64845

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¥
ª
7__inference_batch_normalization_191_layer_call_fn_64065

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_540142
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65353

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_207_layer_call_fn_65417

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_207_layer_call_and_return_conditional_losses_555092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_202_layer_call_and_return_conditional_losses_64668

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¥
ª
7__inference_batch_normalization_192_layer_call_fn_64213

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_541182
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_53910

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64419

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
L
0__inference_max_pooling2d_10_layer_call_fn_54685

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_546792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
ª
7__inference_batch_normalization_197_layer_call_fn_64940

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_560912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs


*__inference_conv2d_205_layer_call_fn_65121

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_205_layer_call_and_return_conditional_losses_557822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_204_layer_call_and_return_conditional_losses_64964

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_54998

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_53983

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
£
ª
7__inference_batch_normalization_196_layer_call_fn_64728

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_545272
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_56091

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_54087

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_64993

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Å
rcnn_1_map_while_cond_529552
.rcnn_1_map_while_rcnn_1_map_while_loop_counter-
)rcnn_1_map_while_rcnn_1_map_strided_slice 
rcnn_1_map_while_placeholder"
rcnn_1_map_while_placeholder_12
.rcnn_1_map_while_less_rcnn_1_map_strided_sliceI
Ercnn_1_map_while_rcnn_1_map_while_cond_52955___redundant_placeholder0I
Ercnn_1_map_while_rcnn_1_map_while_cond_52955___redundant_placeholder1	I
Ercnn_1_map_while_rcnn_1_map_while_cond_52955___redundant_placeholder2	I
Ercnn_1_map_while_rcnn_1_map_while_cond_52955___redundant_placeholder3
rcnn_1_map_while_identity
¥
rcnn_1/map/while/LessLessrcnn_1_map_while_placeholder.rcnn_1_map_while_less_rcnn_1_map_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map/while/Less¶
rcnn_1/map/while/Less_1Less.rcnn_1_map_while_rcnn_1_map_while_loop_counter)rcnn_1_map_while_rcnn_1_map_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map/while/Less_1
rcnn_1/map/while/LogicalAnd
LogicalAndrcnn_1/map/while/Less_1:z:0rcnn_1/map/while/Less:z:0*
_output_shapes
: 2
rcnn_1/map/while/LogicalAnd
rcnn_1/map/while/IdentityIdentityrcnn_1/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
rcnn_1/map/while/Identity"?
rcnn_1_map_while_identity"rcnn_1/map/while/Identity:output:0*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
Ý

R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_63975

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ý
ª
7__inference_batch_normalization_191_layer_call_fn_64001

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_552862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_201_layer_call_and_return_conditional_losses_64520

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ª
7__inference_batch_normalization_197_layer_call_fn_64876

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_546312
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_56657

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
°
ª
B__inference_dense_9_layer_call_and_return_conditional_losses_65627

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_55835

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64697

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º"
À
rcnn_1_map_while_body_529562
.rcnn_1_map_while_rcnn_1_map_while_loop_counter-
)rcnn_1_map_while_rcnn_1_map_strided_slice 
rcnn_1_map_while_placeholder"
rcnn_1_map_while_placeholder_11
-rcnn_1_map_while_rcnn_1_map_strided_slice_1_0m
ircnn_1_map_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_tensorarrayunstack_tensorlistfromtensor_0
rcnn_1_map_while_53047_0	
rcnn_1_map_while_53049_0	!
rcnn_1_map_while_rcnn_1_pad_0
rcnn_1_map_while_identity
rcnn_1_map_while_identity_1
rcnn_1_map_while_identity_2
rcnn_1_map_while_identity_3/
+rcnn_1_map_while_rcnn_1_map_strided_slice_1k
grcnn_1_map_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_tensorarrayunstack_tensorlistfromtensor
rcnn_1_map_while_53047	
rcnn_1_map_while_53049	
rcnn_1_map_while_rcnn_1_padÒ
Brcnn_1/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Brcnn_1/map/while/TensorArrayV2Read/TensorListGetItem/element_shape
4rcnn_1/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemircnn_1_map_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_tensorarrayunstack_tensorlistfromtensor_0rcnn_1_map_while_placeholderKrcnn_1/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:*
element_dtype0	26
4rcnn_1/map/while/TensorArrayV2Read/TensorListGetItemÙ
 rcnn_1/map/while/PartitionedCallPartitionedCall;rcnn_1/map/while/TensorArrayV2Read/TensorListGetItem:item:0rcnn_1_map_while_53047_0rcnn_1_map_while_53049_0rcnn_1_map_while_rcnn_1_pad_0*
Tin
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_getROIfeature2_530462"
 rcnn_1/map/while/PartitionedCall
5rcnn_1/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrcnn_1_map_while_placeholder_1rcnn_1_map_while_placeholder)rcnn_1/map/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype027
5rcnn_1/map/while/TensorArrayV2Write/TensorListSetItemr
rcnn_1/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map/while/add/y
rcnn_1/map/while/addAddV2rcnn_1_map_while_placeholderrcnn_1/map/while/add/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map/while/addv
rcnn_1/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map/while/add_1/y­
rcnn_1/map/while/add_1AddV2.rcnn_1_map_while_rcnn_1_map_while_loop_counter!rcnn_1/map/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map/while/add_1
rcnn_1/map/while/IdentityIdentityrcnn_1/map/while/add_1:z:0*
T0*
_output_shapes
: 2
rcnn_1/map/while/Identity
rcnn_1/map/while/Identity_1Identity)rcnn_1_map_while_rcnn_1_map_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map/while/Identity_1
rcnn_1/map/while/Identity_2Identityrcnn_1/map/while/add:z:0*
T0*
_output_shapes
: 2
rcnn_1/map/while/Identity_2®
rcnn_1/map/while/Identity_3IdentityErcnn_1/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
rcnn_1/map/while/Identity_3"2
rcnn_1_map_while_53047rcnn_1_map_while_53047_0"2
rcnn_1_map_while_53049rcnn_1_map_while_53049_0"?
rcnn_1_map_while_identity"rcnn_1/map/while/Identity:output:0"C
rcnn_1_map_while_identity_1$rcnn_1/map/while/Identity_1:output:0"C
rcnn_1_map_while_identity_2$rcnn_1/map/while/Identity_2:output:0"C
rcnn_1_map_while_identity_3$rcnn_1/map/while/Identity_3:output:0"\
+rcnn_1_map_while_rcnn_1_map_strided_slice_1-rcnn_1_map_while_rcnn_1_map_strided_slice_1_0"<
rcnn_1_map_while_rcnn_1_padrcnn_1_map_while_rcnn_1_pad_0"Ô
grcnn_1_map_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_tensorarrayunstack_tensorlistfromtensorircnn_1_map_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_tensorarrayunstack_tensorlistfromtensor_0*G
_input_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
	
­
E__inference_conv2d_202_layer_call_and_return_conditional_losses_55956

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Û
ª
7__inference_batch_normalization_200_layer_call_fn_65384

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_554442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_56009

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_203_layer_call_and_return_conditional_losses_64816

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
h
 __inference_getROIfeature5_53303

inputs	
unknown	
	unknown_0	

rcnn_1_pad
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constß
PartitionedCallPartitionedCallinputsConst:output:0unknown	unknown_0
rcnn_1_pad*
Tin	
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference_getROIfeature_530382
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.::::ÿÿÿÿÿÿÿÿÿÀ:B >

_output_shapes
:
 
_user_specified_nameinputs:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
Û
ª
7__inference_batch_normalization_196_layer_call_fn_64792

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_559912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
ª
7__inference_batch_normalization_201_layer_call_fn_65545

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_555622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65141

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
£
ª
7__inference_batch_normalization_198_layer_call_fn_65024

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_547472
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_6_layer_call_and_return_conditional_losses_56725

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64123

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
«
C__inference_dense_11_layer_call_and_return_conditional_losses_65666

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

é
rcnn_1_map_4_while_cond_535266
2rcnn_1_map_4_while_rcnn_1_map_4_while_loop_counter1
-rcnn_1_map_4_while_rcnn_1_map_4_strided_slice"
rcnn_1_map_4_while_placeholder$
 rcnn_1_map_4_while_placeholder_16
2rcnn_1_map_4_while_less_rcnn_1_map_4_strided_sliceM
Ircnn_1_map_4_while_rcnn_1_map_4_while_cond_53526___redundant_placeholder0M
Ircnn_1_map_4_while_rcnn_1_map_4_while_cond_53526___redundant_placeholder1	M
Ircnn_1_map_4_while_rcnn_1_map_4_while_cond_53526___redundant_placeholder2	M
Ircnn_1_map_4_while_rcnn_1_map_4_while_cond_53526___redundant_placeholder3
rcnn_1_map_4_while_identity
¯
rcnn_1/map_4/while/LessLessrcnn_1_map_4_while_placeholder2rcnn_1_map_4_while_less_rcnn_1_map_4_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_4/while/LessÂ
rcnn_1/map_4/while/Less_1Less2rcnn_1_map_4_while_rcnn_1_map_4_while_loop_counter-rcnn_1_map_4_while_rcnn_1_map_4_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_4/while/Less_1 
rcnn_1/map_4/while/LogicalAnd
LogicalAndrcnn_1/map_4/while/Less_1:z:0rcnn_1/map_4/while/Less:z:0*
_output_shapes
: 2
rcnn_1/map_4/while/LogicalAnd
rcnn_1/map_4/while/IdentityIdentity!rcnn_1/map_4/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
rcnn_1/map_4/while/Identity"C
rcnn_1_map_4_while_identity$rcnn_1/map_4/while/Identity:output:0*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
¦

R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65307

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_55562

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_55991

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8

__inference_getROIfeature_53038

inputs	
size
strided_slice_3_input	
strided_slice_5_input	
strided_slice_7_rcnn_1_pad
identityt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ú
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ä
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ä
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yJ
addAddV2sizeadd/y:output:0*
T0*
_output_shapes
: 2
addj
strided_slice_3/stackPacksize*
N*
T0*
_output_shapes
:2
strided_slice_3/stackq
strided_slice_3/stack_1Packadd:z:0*
N*
T0*
_output_shapes
:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ó
strided_slice_3StridedSlicestrided_slice_3_inputstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3f
mulMulstrided_slice_1:output:0strided_slice_3:output:0*
T0	*
_output_shapes
: 2
mulT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yP
add_1AddV2sizeadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1j
strided_slice_4/stackPacksize*
N*
T0*
_output_shapes
:2
strided_slice_4/stacks
strided_slice_4/stack_1Pack	add_1:z:0*
N*
T0*
_output_shapes
:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2ó
strided_slice_4StridedSlicestrided_slice_3_inputstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_slice_4j
mul_1Mulstrided_slice_2:output:0strided_slice_4:output:0*
T0	*
_output_shapes
: 2
mul_1T
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yP
add_2AddV2sizeadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2j
strided_slice_5/stackPacksize*
N*
T0*
_output_shapes
:2
strided_slice_5/stacks
strided_slice_5/stack_1Pack	add_2:z:0*
N*
T0*
_output_shapes
:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2ó
strided_slice_5StridedSlicestrided_slice_5_inputstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_slice_5[
add_3AddV2mul:z:0strided_slice_5:output:0*
T0	*
_output_shapes
: 2
add_3T
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yP
add_4AddV2sizeadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4j
strided_slice_6/stackPacksize*
N*
T0*
_output_shapes
:2
strided_slice_6/stacks
strided_slice_6/stack_1Pack	add_4:z:0*
N*
T0*
_output_shapes
:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2ó
strided_slice_6StridedSlicestrided_slice_5_inputstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2
strided_slice_6]
add_5AddV2	mul_1:z:0strided_slice_6:output:0*
T0	*
_output_shapes
: 2
add_5T
add_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2	
add_6/yb
add_6AddV2strided_slice:output:0add_6/y:output:0*
T0	*
_output_shapes
: 2
add_6t
strided_slice_7/stack/3Const*
_output_shapes
: *
dtype0	*
value	B	 R 2
strided_slice_7/stack/3²
strided_slice_7/stackPackstrided_slice:output:0mul:z:0	mul_1:z:0 strided_slice_7/stack/3:output:0*
N*
T0	*
_output_shapes
:2
strided_slice_7/stackx
strided_slice_7/stack_1/3Const*
_output_shapes
: *
dtype0	*
value	B	 R 2
strided_slice_7/stack_1/3­
strided_slice_7/stack_1Pack	add_6:z:0	add_3:z:0	add_5:z:0"strided_slice_7/stack_1/3:output:0*
N*
T0	*
_output_shapes
:2
strided_slice_7/stack_1
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_7/stack_2
strided_slice_7/CastCast strided_slice_7/stack_2:output:0*

DstT0	*

SrcT0*
_output_shapes
:2
strided_slice_7/Cast±
strided_slice_7StridedSlicestrided_slice_7_rcnn_1_padstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0strided_slice_7/Cast:y:0*
Index0	*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7z
IdentityIdentitystrided_slice_7:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:: :::ÿÿÿÿÿÿÿÿÿÀ:B >

_output_shapes
:
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namesize:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
£
ª
7__inference_batch_normalization_192_layer_call_fn_64200

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_540872
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_199_layer_call_and_return_conditional_losses_64224

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:		0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_200_layer_call_and_return_conditional_losses_56230

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
×
ª
7__inference_batch_normalization_193_layer_call_fn_64348

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_566392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_56109

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
	
­
E__inference_conv2d_206_layer_call_and_return_conditional_losses_55409

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs


R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_54882

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs


*__inference_conv2d_206_layer_call_fn_65269

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_206_layer_call_and_return_conditional_losses_554092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¯
ª
B__inference_dense_8_layer_call_and_return_conditional_losses_56803

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
|
'__inference_dense_6_layer_call_fn_65576

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_567252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
£
ª
7__inference_batch_normalization_194_layer_call_fn_64432

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_543072
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_64021

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
ª
ª
B__inference_dense_7_layer_call_and_return_conditional_losses_56764

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_55102

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65011

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_54899

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65205

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
£
ª
7__inference_batch_normalization_191_layer_call_fn_64052

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_539832
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
ª
ª
B__inference_dense_7_layer_call_and_return_conditional_losses_65587

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
ª
7__inference_batch_normalization_201_layer_call_fn_65532

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_555442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65075

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_199_layer_call_and_return_conditional_losses_56604

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:		0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_198_layer_call_fn_64085

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_198_layer_call_and_return_conditional_losses_565042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_55462

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
ª
7__inference_batch_normalization_197_layer_call_fn_64953

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_561092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
	
­
E__inference_conv2d_198_layer_call_and_return_conditional_losses_56504

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

é
rcnn_1_map_3_while_cond_533986
2rcnn_1_map_3_while_rcnn_1_map_3_while_loop_counter1
-rcnn_1_map_3_while_rcnn_1_map_3_strided_slice"
rcnn_1_map_3_while_placeholder$
 rcnn_1_map_3_while_placeholder_16
2rcnn_1_map_3_while_less_rcnn_1_map_3_strided_sliceM
Ircnn_1_map_3_while_rcnn_1_map_3_while_cond_53398___redundant_placeholder0M
Ircnn_1_map_3_while_rcnn_1_map_3_while_cond_53398___redundant_placeholder1	M
Ircnn_1_map_3_while_rcnn_1_map_3_while_cond_53398___redundant_placeholder2	M
Ircnn_1_map_3_while_rcnn_1_map_3_while_cond_53398___redundant_placeholder3
rcnn_1_map_3_while_identity
¯
rcnn_1/map_3/while/LessLessrcnn_1_map_3_while_placeholder2rcnn_1_map_3_while_less_rcnn_1_map_3_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_3/while/LessÂ
rcnn_1/map_3/while/Less_1Less2rcnn_1_map_3_while_rcnn_1_map_3_while_loop_counter-rcnn_1_map_3_while_rcnn_1_map_3_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_3/while/Less_1 
rcnn_1/map_3/while/LogicalAnd
LogicalAndrcnn_1/map_3/while/Less_1:z:0rcnn_1/map_3/while/Less:z:0*
_output_shapes
: 2
rcnn_1/map_3/while/LogicalAnd
rcnn_1/map_3/while/IdentityIdentity!rcnn_1/map_3/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
rcnn_1/map_3/while/Identity"C
rcnn_1_map_3_while_identity$rcnn_1/map_3/while/Identity:output:0*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
Ý

R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_55186

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â©
Ñ+
__inference__traced_save_65971
file_prefix7
3savev2_rcnn_1_conv2d_196_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_196_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_190_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_190_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_190_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_190_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_197_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_197_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_191_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_191_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_191_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_191_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_198_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_198_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_192_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_192_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_192_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_192_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_199_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_199_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_193_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_193_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_193_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_193_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_200_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_200_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_194_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_194_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_194_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_194_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_201_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_201_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_195_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_195_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_195_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_195_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_202_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_202_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_196_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_196_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_196_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_196_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_203_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_203_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_197_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_197_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_197_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_197_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_204_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_204_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_198_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_198_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_198_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_198_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_205_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_205_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_199_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_199_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_199_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_199_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_206_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_206_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_200_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_200_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_200_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_200_moving_variance_read_readvariableop7
3savev2_rcnn_1_conv2d_207_kernel_read_readvariableop5
1savev2_rcnn_1_conv2d_207_bias_read_readvariableopC
?savev2_rcnn_1_batch_normalization_201_gamma_read_readvariableopB
>savev2_rcnn_1_batch_normalization_201_beta_read_readvariableopI
Esavev2_rcnn_1_batch_normalization_201_moving_mean_read_readvariableopM
Isavev2_rcnn_1_batch_normalization_201_moving_variance_read_readvariableop4
0savev2_rcnn_1_dense_6_kernel_read_readvariableop2
.savev2_rcnn_1_dense_6_bias_read_readvariableop4
0savev2_rcnn_1_dense_7_kernel_read_readvariableop2
.savev2_rcnn_1_dense_7_bias_read_readvariableop4
0savev2_rcnn_1_dense_8_kernel_read_readvariableop2
.savev2_rcnn_1_dense_8_bias_read_readvariableop4
0savev2_rcnn_1_dense_9_kernel_read_readvariableop2
.savev2_rcnn_1_dense_9_bias_read_readvariableop5
1savev2_rcnn_1_dense_10_kernel_read_readvariableop3
/savev2_rcnn_1_dense_10_bias_read_readvariableop5
1savev2_rcnn_1_dense_11_kernel_read_readvariableop3
/savev2_rcnn_1_dense_11_bias_read_readvariableop
savev2_const_2

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d8c6c5996ead489c87fe5a2dead4a042/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*«$
value¡$B$UB2conv2d_condense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB0conv2d_condense1/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB2conv2d_condense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB0conv2d_condense2/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB;conv2d_extract12_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB9conv2d_extract12_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm_extract12a/gamma/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract12a/beta/.ATTRIBUTES/VARIABLE_VALUEB<batch_norm_extract12a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@batch_norm_extract12a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB2conv2d_extract12/kernel/.ATTRIBUTES/VARIABLE_VALUEB0conv2d_extract12/bias/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm_extract12b/gamma/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract12b/beta/.ATTRIBUTES/VARIABLE_VALUEB<batch_norm_extract12b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@batch_norm_extract12b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB:conv2d_extract8_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB8conv2d_extract8_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract8a/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract8a/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract8a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract8a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1conv2d_extract8/kernel/.ATTRIBUTES/VARIABLE_VALUEB/conv2d_extract8/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract8b/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract8b/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract8b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract8b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB:conv2d_extract5_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB8conv2d_extract5_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract5a/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract5a/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract5a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract5a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1conv2d_extract5/kernel/.ATTRIBUTES/VARIABLE_VALUEB/conv2d_extract5/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract5b/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract5b/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract5b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract5b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB:conv2d_extract3_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB8conv2d_extract3_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract3a/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract3a/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract3a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract3a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1conv2d_extract3/kernel/.ATTRIBUTES/VARIABLE_VALUEB/conv2d_extract3/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract3b/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract3b/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract3b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract3b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB:conv2d_extract2_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB8conv2d_extract2_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract2a/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract2a/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract2a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract2a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1conv2d_extract2/kernel/.ATTRIBUTES/VARIABLE_VALUEB/conv2d_extract2/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract2b/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract2b/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract2b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract2b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB-classifier1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+classifier1/bias/.ATTRIBUTES/VARIABLE_VALUEB-classifier2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+classifier2/bias/.ATTRIBUTES/VARIABLE_VALUEB-classifier3/kernel/.ATTRIBUTES/VARIABLE_VALUEB+classifier3/bias/.ATTRIBUTES/VARIABLE_VALUEB,regressor1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*regressor1/bias/.ATTRIBUTES/VARIABLE_VALUEB,regressor2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*regressor2/bias/.ATTRIBUTES/VARIABLE_VALUEB,regressor3/kernel/.ATTRIBUTES/VARIABLE_VALUEB*regressor3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesµ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*¿
valueµB²UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¤*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_rcnn_1_conv2d_196_kernel_read_readvariableop1savev2_rcnn_1_conv2d_196_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_190_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_190_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_190_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_190_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_197_kernel_read_readvariableop1savev2_rcnn_1_conv2d_197_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_191_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_191_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_191_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_191_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_198_kernel_read_readvariableop1savev2_rcnn_1_conv2d_198_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_192_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_192_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_192_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_192_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_199_kernel_read_readvariableop1savev2_rcnn_1_conv2d_199_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_193_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_193_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_193_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_193_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_200_kernel_read_readvariableop1savev2_rcnn_1_conv2d_200_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_194_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_194_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_194_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_194_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_201_kernel_read_readvariableop1savev2_rcnn_1_conv2d_201_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_195_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_195_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_195_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_195_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_202_kernel_read_readvariableop1savev2_rcnn_1_conv2d_202_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_196_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_196_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_196_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_196_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_203_kernel_read_readvariableop1savev2_rcnn_1_conv2d_203_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_197_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_197_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_197_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_197_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_204_kernel_read_readvariableop1savev2_rcnn_1_conv2d_204_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_198_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_198_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_198_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_198_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_205_kernel_read_readvariableop1savev2_rcnn_1_conv2d_205_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_199_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_199_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_199_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_199_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_206_kernel_read_readvariableop1savev2_rcnn_1_conv2d_206_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_200_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_200_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_200_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_200_moving_variance_read_readvariableop3savev2_rcnn_1_conv2d_207_kernel_read_readvariableop1savev2_rcnn_1_conv2d_207_bias_read_readvariableop?savev2_rcnn_1_batch_normalization_201_gamma_read_readvariableop>savev2_rcnn_1_batch_normalization_201_beta_read_readvariableopEsavev2_rcnn_1_batch_normalization_201_moving_mean_read_readvariableopIsavev2_rcnn_1_batch_normalization_201_moving_variance_read_readvariableop0savev2_rcnn_1_dense_6_kernel_read_readvariableop.savev2_rcnn_1_dense_6_bias_read_readvariableop0savev2_rcnn_1_dense_7_kernel_read_readvariableop.savev2_rcnn_1_dense_7_bias_read_readvariableop0savev2_rcnn_1_dense_8_kernel_read_readvariableop.savev2_rcnn_1_dense_8_bias_read_readvariableop0savev2_rcnn_1_dense_9_kernel_read_readvariableop.savev2_rcnn_1_dense_9_bias_read_readvariableop1savev2_rcnn_1_dense_10_kernel_read_readvariableop/savev2_rcnn_1_dense_10_bias_read_readvariableop1savev2_rcnn_1_dense_11_kernel_read_readvariableop/savev2_rcnn_1_dense_11_bias_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *c
dtypesY
W2U2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ú
_input_shapesè
å: :::::::À:À:À:À:À:À:À::::::		0:0:0:0:0:0:À::::::0:0:0:0:0:0:À::::::0:0:0:0:0:0:À::::::0:0:0:0:0:0:À::::::0:0:0:0:0:0:
À::	@:@:@::
À::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
:À:!

_output_shapes	
:À:!	

_output_shapes	
:À:!


_output_shapes	
:À:!

_output_shapes	
:À:!

_output_shapes	
:À:.*
(
_output_shapes
:À:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
:		0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:.*
(
_output_shapes
:À:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
:0:  

_output_shapes
:0: !

_output_shapes
:0: "

_output_shapes
:0: #

_output_shapes
:0: $

_output_shapes
:0:.%*
(
_output_shapes
:À:!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::!)

_output_shapes	
::!*

_output_shapes	
::-+)
'
_output_shapes
:0: ,

_output_shapes
:0: -

_output_shapes
:0: .

_output_shapes
:0: /

_output_shapes
:0: 0

_output_shapes
:0:.1*
(
_output_shapes
:À:!2

_output_shapes	
::!3

_output_shapes	
::!4

_output_shapes	
::!5

_output_shapes	
::!6

_output_shapes	
::-7)
'
_output_shapes
:0: 8

_output_shapes
:0: 9

_output_shapes
:0: :

_output_shapes
:0: ;

_output_shapes
:0: <

_output_shapes
:0:.=*
(
_output_shapes
:À:!>

_output_shapes	
::!?

_output_shapes	
::!@

_output_shapes	
::!A

_output_shapes	
::!B

_output_shapes	
::-C)
'
_output_shapes
:0: D

_output_shapes
:0: E

_output_shapes
:0: F

_output_shapes
:0: G

_output_shapes
:0: H

_output_shapes
:0:&I"
 
_output_shapes
:
À:!J

_output_shapes	
::%K!

_output_shapes
:	@: L

_output_shapes
:@:$M 

_output_shapes

:@: N

_output_shapes
::&O"
 
_output_shapes
:
À:!P

_output_shapes	
::&Q"
 
_output_shapes
:
:!R

_output_shapes	
::%S!

_output_shapes
:	: T

_output_shapes
::U

_output_shapes
: 

¯
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64909

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¥
ª
7__inference_batch_normalization_200_layer_call_fn_65333

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_549982
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
}
(__inference_dense_11_layer_call_fn_65675

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_569192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
E
)__inference_flatten_1_layer_call_fn_65556

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_567022
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63827

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ:
à
#__inference_signature_wrapper_59613
input_1
	input_2_1
	input_2_2
	input_2_3
	input_2_4
	input_2_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11	

unknown_12	

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62

unknown_63

unknown_64

unknown_65

unknown_66

unknown_67

unknown_68

unknown_69

unknown_70

unknown_71

unknown_72

unknown_73

unknown_74

unknown_75

unknown_76

unknown_77

unknown_78

unknown_79

unknown_80

unknown_81

unknown_82

unknown_83

unknown_84
identity

identity_1

identity_2	

identity_3

identity_4

identity_5	

identity_6

identity_7

identity_8	

identity_9
identity_10
identity_11	
identity_12
identity_13
identity_14	¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinput_1	input_2_1	input_2_2	input_2_3	input_2_4	input_2_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84*g
Tin`
^2\		*
Tout
2					*
_collective_manager_ids
 *³
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*v
_read_only_resource_inputsX
VT	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_538172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_5

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_6

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_7

Identity_8Identity StatefulPartitionedCall:output:8^StatefulPartitionedCall*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_8

Identity_9Identity StatefulPartitionedCall:output:9^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_9
Identity_10Identity!StatefulPartitionedCall:output:10^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_10
Identity_11Identity!StatefulPartitionedCall:output:11^StatefulPartitionedCall*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_11
Identity_12Identity!StatefulPartitionedCall:output:12^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_12
Identity_13Identity!StatefulPartitionedCall:output:13^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_13
Identity_14Identity!StatefulPartitionedCall:output:14^StatefulPartitionedCall*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_14"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*
_input_shapes
ÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_1:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_2:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_3:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_4:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_5
Ö
¯
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_54307

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
¾,
 __inference__wrapped_model_53817
input_1
	input_2_1
	input_2_2
	input_2_3
	input_2_4
	input_2_54
0rcnn_1_conv2d_196_conv2d_readvariableop_resource5
1rcnn_1_conv2d_196_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_190_readvariableop_resource<
8rcnn_1_batch_normalization_190_readvariableop_1_resourceK
Grcnn_1_batch_normalization_190_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_190_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_197_conv2d_readvariableop_resource5
1rcnn_1_conv2d_197_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_191_readvariableop_resource<
8rcnn_1_batch_normalization_191_readvariableop_1_resourceK
Grcnn_1_batch_normalization_191_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_191_fusedbatchnormv3_readvariableop_1_resource
rcnn_1_map_while_input_6	
rcnn_1_map_while_input_7	4
0rcnn_1_conv2d_206_conv2d_readvariableop_resource5
1rcnn_1_conv2d_206_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_200_readvariableop_resource<
8rcnn_1_batch_normalization_200_readvariableop_1_resourceK
Grcnn_1_batch_normalization_200_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_200_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_207_conv2d_readvariableop_resource5
1rcnn_1_conv2d_207_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_201_readvariableop_resource<
8rcnn_1_batch_normalization_201_readvariableop_1_resourceK
Grcnn_1_batch_normalization_201_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_201_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_204_conv2d_readvariableop_resource5
1rcnn_1_conv2d_204_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_198_readvariableop_resource<
8rcnn_1_batch_normalization_198_readvariableop_1_resourceK
Grcnn_1_batch_normalization_198_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_198_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_205_conv2d_readvariableop_resource5
1rcnn_1_conv2d_205_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_199_readvariableop_resource<
8rcnn_1_batch_normalization_199_readvariableop_1_resourceK
Grcnn_1_batch_normalization_199_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_199_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_202_conv2d_readvariableop_resource5
1rcnn_1_conv2d_202_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_196_readvariableop_resource<
8rcnn_1_batch_normalization_196_readvariableop_1_resourceK
Grcnn_1_batch_normalization_196_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_196_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_203_conv2d_readvariableop_resource5
1rcnn_1_conv2d_203_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_197_readvariableop_resource<
8rcnn_1_batch_normalization_197_readvariableop_1_resourceK
Grcnn_1_batch_normalization_197_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_197_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_200_conv2d_readvariableop_resource5
1rcnn_1_conv2d_200_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_194_readvariableop_resource<
8rcnn_1_batch_normalization_194_readvariableop_1_resourceK
Grcnn_1_batch_normalization_194_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_194_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_201_conv2d_readvariableop_resource5
1rcnn_1_conv2d_201_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_195_readvariableop_resource<
8rcnn_1_batch_normalization_195_readvariableop_1_resourceK
Grcnn_1_batch_normalization_195_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_195_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_198_conv2d_readvariableop_resource5
1rcnn_1_conv2d_198_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_192_readvariableop_resource<
8rcnn_1_batch_normalization_192_readvariableop_1_resourceK
Grcnn_1_batch_normalization_192_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_192_fusedbatchnormv3_readvariableop_1_resource4
0rcnn_1_conv2d_199_conv2d_readvariableop_resource5
1rcnn_1_conv2d_199_biasadd_readvariableop_resource:
6rcnn_1_batch_normalization_193_readvariableop_resource<
8rcnn_1_batch_normalization_193_readvariableop_1_resourceK
Grcnn_1_batch_normalization_193_fusedbatchnormv3_readvariableop_resourceM
Ircnn_1_batch_normalization_193_fusedbatchnormv3_readvariableop_1_resource1
-rcnn_1_dense_6_matmul_readvariableop_resource2
.rcnn_1_dense_6_biasadd_readvariableop_resource1
-rcnn_1_dense_7_matmul_readvariableop_resource2
.rcnn_1_dense_7_biasadd_readvariableop_resource1
-rcnn_1_dense_8_matmul_readvariableop_resource2
.rcnn_1_dense_8_biasadd_readvariableop_resource1
-rcnn_1_dense_9_matmul_readvariableop_resource2
.rcnn_1_dense_9_biasadd_readvariableop_resource2
.rcnn_1_dense_10_matmul_readvariableop_resource3
/rcnn_1_dense_10_biasadd_readvariableop_resource2
.rcnn_1_dense_11_matmul_readvariableop_resource3
/rcnn_1_dense_11_biasadd_readvariableop_resource
identity

identity_1

identity_2	

identity_3

identity_4

identity_5	

identity_6

identity_7

identity_8	

identity_9
identity_10
identity_11	
identity_12
identity_13
identity_14	Í
'rcnn_1/conv2d_196/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_196_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'rcnn_1/conv2d_196/Conv2D/ReadVariableOpÛ
rcnn_1/conv2d_196/Conv2DConv2Dinput_1/rcnn_1/conv2d_196/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
rcnn_1/conv2d_196/Conv2DÃ
(rcnn_1/conv2d_196/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_196_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/conv2d_196/BiasAdd/ReadVariableOpÑ
rcnn_1/conv2d_196/BiasAddBiasAdd!rcnn_1/conv2d_196/Conv2D:output:00rcnn_1/conv2d_196/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_196/BiasAdd
rcnn_1/conv2d_196/ReluRelu"rcnn_1/conv2d_196/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_196/ReluÒ
-rcnn_1/batch_normalization_190/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_190_readvariableop_resource*
_output_shapes	
:*
dtype02/
-rcnn_1/batch_normalization_190/ReadVariableOpØ
/rcnn_1/batch_normalization_190/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_190_readvariableop_1_resource*
_output_shapes	
:*
dtype021
/rcnn_1/batch_normalization_190/ReadVariableOp_1
>rcnn_1/batch_normalization_190/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_190_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02@
>rcnn_1/batch_normalization_190/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_190/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_190_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02B
@rcnn_1/batch_normalization_190/FusedBatchNormV3/ReadVariableOp_1§
/rcnn_1/batch_normalization_190/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_196/Relu:activations:05rcnn_1/batch_normalization_190/ReadVariableOp:value:07rcnn_1/batch_normalization_190/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_190/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_190/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_190/FusedBatchNormV3Í
'rcnn_1/conv2d_197/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_197_conv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02)
'rcnn_1/conv2d_197/Conv2D/ReadVariableOp
rcnn_1/conv2d_197/Conv2DConv2D3rcnn_1/batch_normalization_190/FusedBatchNormV3:y:0/rcnn_1/conv2d_197/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingSAME*
strides
2
rcnn_1/conv2d_197/Conv2DÃ
(rcnn_1/conv2d_197/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_197_biasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02*
(rcnn_1/conv2d_197/BiasAdd/ReadVariableOpÑ
rcnn_1/conv2d_197/BiasAddBiasAdd!rcnn_1/conv2d_197/Conv2D:output:00rcnn_1/conv2d_197/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/conv2d_197/BiasAdd
rcnn_1/conv2d_197/ReluRelu"rcnn_1/conv2d_197/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/conv2d_197/ReluÒ
-rcnn_1/batch_normalization_191/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_191_readvariableop_resource*
_output_shapes	
:À*
dtype02/
-rcnn_1/batch_normalization_191/ReadVariableOpØ
/rcnn_1/batch_normalization_191/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_191_readvariableop_1_resource*
_output_shapes	
:À*
dtype021
/rcnn_1/batch_normalization_191/ReadVariableOp_1
>rcnn_1/batch_normalization_191/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_191_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02@
>rcnn_1/batch_normalization_191/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_191/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_191_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02B
@rcnn_1/batch_normalization_191/FusedBatchNormV3/ReadVariableOp_1§
/rcnn_1/batch_normalization_191/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_197/Relu:activations:05rcnn_1/batch_normalization_191/ReadVariableOp:value:07rcnn_1/batch_normalization_191/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_191/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_191/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_191/FusedBatchNormV3
rcnn_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2
rcnn_1/Pad/paddings­

rcnn_1/PadPad3rcnn_1/batch_normalization_191/FusedBatchNormV3:y:0rcnn_1/Pad/paddings:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

rcnn_1/Pad
rcnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
rcnn_1/strided_slice/stack
rcnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
rcnn_1/strided_slice/stack_1
rcnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
rcnn_1/strided_slice/stack_2·
rcnn_1/strided_sliceStridedSlice	input_2_1#rcnn_1/strided_slice/stack:output:0%rcnn_1/strided_slice/stack_1:output:0%rcnn_1/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask2
rcnn_1/strided_slicei
rcnn_1/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
rcnn_1/Greater/y
rcnn_1/GreaterGreaterrcnn_1/strided_slice:output:0rcnn_1/Greater/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Greaterb
rcnn_1/WhereWherercnn_1/Greater:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Whereh
rcnn_1/map/ShapeShapercnn_1/Where:index:0*
T0	*
_output_shapes
:2
rcnn_1/map/Shape
rcnn_1/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
rcnn_1/map/strided_slice/stack
 rcnn_1/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 rcnn_1/map/strided_slice/stack_1
 rcnn_1/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 rcnn_1/map/strided_slice/stack_2¤
rcnn_1/map/strided_sliceStridedSlicercnn_1/map/Shape:output:0'rcnn_1/map/strided_slice/stack:output:0)rcnn_1/map/strided_slice/stack_1:output:0)rcnn_1/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rcnn_1/map/strided_slice
&rcnn_1/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&rcnn_1/map/TensorArrayV2/element_shapeÜ
rcnn_1/map/TensorArrayV2TensorListReserve/rcnn_1/map/TensorArrayV2/element_shape:output:0!rcnn_1/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type02
rcnn_1/map/TensorArrayV2Î
@rcnn_1/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2B
@rcnn_1/map/TensorArrayUnstack/TensorListFromTensor/element_shape 
2rcnn_1/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrcnn_1/Where:index:0Ircnn_1/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0	*

shape_type024
2rcnn_1/map/TensorArrayUnstack/TensorListFromTensorf
rcnn_1/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
rcnn_1/map/Const©
(rcnn_1/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2*
(rcnn_1/map/TensorArrayV2_1/element_shapeâ
rcnn_1/map/TensorArrayV2_1TensorListReserve1rcnn_1/map/TensorArrayV2_1/element_shape:output:0!rcnn_1/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rcnn_1/map/TensorArrayV2_1
rcnn_1/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rcnn_1/map/while/loop_counterÊ
rcnn_1/map/whileStatelessWhile&rcnn_1/map/while/loop_counter:output:0!rcnn_1/map/strided_slice:output:0rcnn_1/map/Const:output:0#rcnn_1/map/TensorArrayV2_1:handle:0!rcnn_1/map/strided_slice:output:0Brcnn_1/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0rcnn_1_map_while_input_6rcnn_1_map_while_input_7rcnn_1/Pad:output:0*
T
2			*
_lower_using_switch_merge(*
_num_original_outputs	*H
_output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *'
bodyR
rcnn_1_map_while_body_52956*'
condR
rcnn_1_map_while_cond_52955*G
output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/map/whileÏ
;rcnn_1/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2=
;rcnn_1/map/TensorArrayV2Stack/TensorListStack/element_shape
-rcnn_1/map/TensorArrayV2Stack/TensorListStackTensorListStackrcnn_1/map/while:output:3Drcnn_1/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
element_dtype02/
-rcnn_1/map/TensorArrayV2Stack/TensorListStackÍ
'rcnn_1/conv2d_206/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_206_conv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02)
'rcnn_1/conv2d_206/Conv2D/ReadVariableOp
rcnn_1/conv2d_206/Conv2DConv2D6rcnn_1/map/TensorArrayV2Stack/TensorListStack:tensor:0/rcnn_1/conv2d_206/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
rcnn_1/conv2d_206/Conv2DÃ
(rcnn_1/conv2d_206/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_206_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/conv2d_206/BiasAdd/ReadVariableOpÑ
rcnn_1/conv2d_206/BiasAddBiasAdd!rcnn_1/conv2d_206/Conv2D:output:00rcnn_1/conv2d_206/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_206/BiasAdd
rcnn_1/conv2d_206/ReluRelu"rcnn_1/conv2d_206/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_206/ReluÒ
-rcnn_1/batch_normalization_200/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_200_readvariableop_resource*
_output_shapes	
:*
dtype02/
-rcnn_1/batch_normalization_200/ReadVariableOpØ
/rcnn_1/batch_normalization_200/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_200_readvariableop_1_resource*
_output_shapes	
:*
dtype021
/rcnn_1/batch_normalization_200/ReadVariableOp_1
>rcnn_1/batch_normalization_200/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_200_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02@
>rcnn_1/batch_normalization_200/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_200/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_200_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02B
@rcnn_1/batch_normalization_200/FusedBatchNormV3/ReadVariableOp_1§
/rcnn_1/batch_normalization_200/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_206/Relu:activations:05rcnn_1/batch_normalization_200/ReadVariableOp:value:07rcnn_1/batch_normalization_200/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_200/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_200/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_200/FusedBatchNormV3Ì
'rcnn_1/conv2d_207/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_207_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02)
'rcnn_1/conv2d_207/Conv2D/ReadVariableOp
rcnn_1/conv2d_207/Conv2DConv2D3rcnn_1/batch_normalization_200/FusedBatchNormV3:y:0/rcnn_1/conv2d_207/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
rcnn_1/conv2d_207/Conv2DÂ
(rcnn_1/conv2d_207/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_207_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(rcnn_1/conv2d_207/BiasAdd/ReadVariableOpÐ
rcnn_1/conv2d_207/BiasAddBiasAdd!rcnn_1/conv2d_207/Conv2D:output:00rcnn_1/conv2d_207/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_207/BiasAdd
rcnn_1/conv2d_207/ReluRelu"rcnn_1/conv2d_207/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_207/ReluÑ
-rcnn_1/batch_normalization_201/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_201_readvariableop_resource*
_output_shapes
:0*
dtype02/
-rcnn_1/batch_normalization_201/ReadVariableOp×
/rcnn_1/batch_normalization_201/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_201_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/rcnn_1/batch_normalization_201/ReadVariableOp_1
>rcnn_1/batch_normalization_201/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_201_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>rcnn_1/batch_normalization_201/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_201/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_201_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@rcnn_1/batch_normalization_201/FusedBatchNormV3/ReadVariableOp_1¢
/rcnn_1/batch_normalization_201/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_207/Relu:activations:05rcnn_1/batch_normalization_201/ReadVariableOp:value:07rcnn_1/batch_normalization_201/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_201/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_201/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_201/FusedBatchNormV3s
rcnn_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rcnn_1/concat/axisõ
rcnn_1/concatConcatV23rcnn_1/batch_normalization_200/FusedBatchNormV3:y:03rcnn_1/batch_normalization_201/FusedBatchNormV3:y:0rcnn_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/concat
rcnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
rcnn_1/strided_slice_1/stack
rcnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2 
rcnn_1/strided_slice_1/stack_1
rcnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
rcnn_1/strided_slice_1/stack_2Á
rcnn_1/strided_slice_1StridedSlice	input_2_2%rcnn_1/strided_slice_1/stack:output:0'rcnn_1/strided_slice_1/stack_1:output:0'rcnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask2
rcnn_1/strided_slice_1m
rcnn_1/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
rcnn_1/Greater_1/y£
rcnn_1/Greater_1Greaterrcnn_1/strided_slice_1:output:0rcnn_1/Greater_1/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Greater_1h
rcnn_1/Where_1Wherercnn_1/Greater_1:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Where_1n
rcnn_1/map_1/ShapeShapercnn_1/Where_1:index:0*
T0	*
_output_shapes
:2
rcnn_1/map_1/Shape
 rcnn_1/map_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 rcnn_1/map_1/strided_slice/stack
"rcnn_1/map_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"rcnn_1/map_1/strided_slice/stack_1
"rcnn_1/map_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rcnn_1/map_1/strided_slice/stack_2°
rcnn_1/map_1/strided_sliceStridedSlicercnn_1/map_1/Shape:output:0)rcnn_1/map_1/strided_slice/stack:output:0+rcnn_1/map_1/strided_slice/stack_1:output:0+rcnn_1/map_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rcnn_1/map_1/strided_slice
(rcnn_1/map_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(rcnn_1/map_1/TensorArrayV2/element_shapeä
rcnn_1/map_1/TensorArrayV2TensorListReserve1rcnn_1/map_1/TensorArrayV2/element_shape:output:0#rcnn_1/map_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type02
rcnn_1/map_1/TensorArrayV2Ò
Brcnn_1/map_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Brcnn_1/map_1/TensorArrayUnstack/TensorListFromTensor/element_shape¨
4rcnn_1/map_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrcnn_1/Where_1:index:0Krcnn_1/map_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0	*

shape_type026
4rcnn_1/map_1/TensorArrayUnstack/TensorListFromTensorj
rcnn_1/map_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
rcnn_1/map_1/Const­
*rcnn_1/map_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2,
*rcnn_1/map_1/TensorArrayV2_1/element_shapeê
rcnn_1/map_1/TensorArrayV2_1TensorListReserve3rcnn_1/map_1/TensorArrayV2_1/element_shape:output:0#rcnn_1/map_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rcnn_1/map_1/TensorArrayV2_1
rcnn_1/map_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
rcnn_1/map_1/while/loop_counterÞ
rcnn_1/map_1/whileStatelessWhile(rcnn_1/map_1/while/loop_counter:output:0#rcnn_1/map_1/strided_slice:output:0rcnn_1/map_1/Const:output:0%rcnn_1/map_1/TensorArrayV2_1:handle:0#rcnn_1/map_1/strided_slice:output:0Drcnn_1/map_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0rcnn_1_map_while_input_6rcnn_1_map_while_input_7rcnn_1/Pad:output:0*
T
2			*
_lower_using_switch_merge(*
_num_original_outputs	*H
_output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *)
body!R
rcnn_1_map_1_while_body_53143*)
cond!R
rcnn_1_map_1_while_cond_53142*G
output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/map_1/whileÓ
=rcnn_1/map_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2?
=rcnn_1/map_1/TensorArrayV2Stack/TensorListStack/element_shape¡
/rcnn_1/map_1/TensorArrayV2Stack/TensorListStackTensorListStackrcnn_1/map_1/while:output:3Frcnn_1/map_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
element_dtype021
/rcnn_1/map_1/TensorArrayV2Stack/TensorListStackÍ
'rcnn_1/conv2d_204/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_204_conv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02)
'rcnn_1/conv2d_204/Conv2D/ReadVariableOp
rcnn_1/conv2d_204/Conv2DConv2D8rcnn_1/map_1/TensorArrayV2Stack/TensorListStack:tensor:0/rcnn_1/conv2d_204/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
rcnn_1/conv2d_204/Conv2DÃ
(rcnn_1/conv2d_204/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_204_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/conv2d_204/BiasAdd/ReadVariableOpÑ
rcnn_1/conv2d_204/BiasAddBiasAdd!rcnn_1/conv2d_204/Conv2D:output:00rcnn_1/conv2d_204/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_204/BiasAdd
rcnn_1/conv2d_204/ReluRelu"rcnn_1/conv2d_204/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_204/ReluÒ
-rcnn_1/batch_normalization_198/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_198_readvariableop_resource*
_output_shapes	
:*
dtype02/
-rcnn_1/batch_normalization_198/ReadVariableOpØ
/rcnn_1/batch_normalization_198/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_198_readvariableop_1_resource*
_output_shapes	
:*
dtype021
/rcnn_1/batch_normalization_198/ReadVariableOp_1
>rcnn_1/batch_normalization_198/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_198_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02@
>rcnn_1/batch_normalization_198/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_198/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_198_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02B
@rcnn_1/batch_normalization_198/FusedBatchNormV3/ReadVariableOp_1§
/rcnn_1/batch_normalization_198/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_204/Relu:activations:05rcnn_1/batch_normalization_198/ReadVariableOp:value:07rcnn_1/batch_normalization_198/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_198/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_198/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_198/FusedBatchNormV3Ì
'rcnn_1/conv2d_205/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_205_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02)
'rcnn_1/conv2d_205/Conv2D/ReadVariableOp
rcnn_1/conv2d_205/Conv2DConv2D3rcnn_1/batch_normalization_198/FusedBatchNormV3:y:0/rcnn_1/conv2d_205/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
rcnn_1/conv2d_205/Conv2DÂ
(rcnn_1/conv2d_205/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_205_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(rcnn_1/conv2d_205/BiasAdd/ReadVariableOpÐ
rcnn_1/conv2d_205/BiasAddBiasAdd!rcnn_1/conv2d_205/Conv2D:output:00rcnn_1/conv2d_205/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_205/BiasAdd
rcnn_1/conv2d_205/ReluRelu"rcnn_1/conv2d_205/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_205/ReluÑ
-rcnn_1/batch_normalization_199/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_199_readvariableop_resource*
_output_shapes
:0*
dtype02/
-rcnn_1/batch_normalization_199/ReadVariableOp×
/rcnn_1/batch_normalization_199/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_199_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/rcnn_1/batch_normalization_199/ReadVariableOp_1
>rcnn_1/batch_normalization_199/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_199_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>rcnn_1/batch_normalization_199/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_199/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_199_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@rcnn_1/batch_normalization_199/FusedBatchNormV3/ReadVariableOp_1¢
/rcnn_1/batch_normalization_199/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_205/Relu:activations:05rcnn_1/batch_normalization_199/ReadVariableOp:value:07rcnn_1/batch_normalization_199/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_199/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_199/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_199/FusedBatchNormV3ð
rcnn_1/max_pooling2d_11/MaxPoolMaxPool3rcnn_1/batch_normalization_198/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2!
rcnn_1/max_pooling2d_11/MaxPoolw
rcnn_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rcnn_1/concat_1/axisð
rcnn_1/concat_1ConcatV2(rcnn_1/max_pooling2d_11/MaxPool:output:03rcnn_1/batch_normalization_199/FusedBatchNormV3:y:0rcnn_1/concat_1/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/concat_1
rcnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
rcnn_1/strided_slice_2/stack
rcnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2 
rcnn_1/strided_slice_2/stack_1
rcnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
rcnn_1/strided_slice_2/stack_2Á
rcnn_1/strided_slice_2StridedSlice	input_2_3%rcnn_1/strided_slice_2/stack:output:0'rcnn_1/strided_slice_2/stack_1:output:0'rcnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask2
rcnn_1/strided_slice_2m
rcnn_1/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
rcnn_1/Greater_2/y£
rcnn_1/Greater_2Greaterrcnn_1/strided_slice_2:output:0rcnn_1/Greater_2/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Greater_2h
rcnn_1/Where_2Wherercnn_1/Greater_2:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Where_2n
rcnn_1/map_2/ShapeShapercnn_1/Where_2:index:0*
T0	*
_output_shapes
:2
rcnn_1/map_2/Shape
 rcnn_1/map_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 rcnn_1/map_2/strided_slice/stack
"rcnn_1/map_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"rcnn_1/map_2/strided_slice/stack_1
"rcnn_1/map_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rcnn_1/map_2/strided_slice/stack_2°
rcnn_1/map_2/strided_sliceStridedSlicercnn_1/map_2/Shape:output:0)rcnn_1/map_2/strided_slice/stack:output:0+rcnn_1/map_2/strided_slice/stack_1:output:0+rcnn_1/map_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rcnn_1/map_2/strided_slice
(rcnn_1/map_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(rcnn_1/map_2/TensorArrayV2/element_shapeä
rcnn_1/map_2/TensorArrayV2TensorListReserve1rcnn_1/map_2/TensorArrayV2/element_shape:output:0#rcnn_1/map_2/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type02
rcnn_1/map_2/TensorArrayV2Ò
Brcnn_1/map_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Brcnn_1/map_2/TensorArrayUnstack/TensorListFromTensor/element_shape¨
4rcnn_1/map_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrcnn_1/Where_2:index:0Krcnn_1/map_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0	*

shape_type026
4rcnn_1/map_2/TensorArrayUnstack/TensorListFromTensorj
rcnn_1/map_2/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
rcnn_1/map_2/Const­
*rcnn_1/map_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2,
*rcnn_1/map_2/TensorArrayV2_1/element_shapeê
rcnn_1/map_2/TensorArrayV2_1TensorListReserve3rcnn_1/map_2/TensorArrayV2_1/element_shape:output:0#rcnn_1/map_2/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rcnn_1/map_2/TensorArrayV2_1
rcnn_1/map_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
rcnn_1/map_2/while/loop_counterÞ
rcnn_1/map_2/whileStatelessWhile(rcnn_1/map_2/while/loop_counter:output:0#rcnn_1/map_2/strided_slice:output:0rcnn_1/map_2/Const:output:0%rcnn_1/map_2/TensorArrayV2_1:handle:0#rcnn_1/map_2/strided_slice:output:0Drcnn_1/map_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0rcnn_1_map_while_input_6rcnn_1_map_while_input_7rcnn_1/Pad:output:0*
T
2			*
_lower_using_switch_merge(*
_num_original_outputs	*H
_output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *)
body!R
rcnn_1_map_2_while_body_53271*)
cond!R
rcnn_1_map_2_while_cond_53270*G
output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/map_2/whileÓ
=rcnn_1/map_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2?
=rcnn_1/map_2/TensorArrayV2Stack/TensorListStack/element_shape¡
/rcnn_1/map_2/TensorArrayV2Stack/TensorListStackTensorListStackrcnn_1/map_2/while:output:3Frcnn_1/map_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
element_dtype021
/rcnn_1/map_2/TensorArrayV2Stack/TensorListStackÍ
'rcnn_1/conv2d_202/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_202_conv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02)
'rcnn_1/conv2d_202/Conv2D/ReadVariableOp
rcnn_1/conv2d_202/Conv2DConv2D8rcnn_1/map_2/TensorArrayV2Stack/TensorListStack:tensor:0/rcnn_1/conv2d_202/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
rcnn_1/conv2d_202/Conv2DÃ
(rcnn_1/conv2d_202/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_202_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/conv2d_202/BiasAdd/ReadVariableOpÑ
rcnn_1/conv2d_202/BiasAddBiasAdd!rcnn_1/conv2d_202/Conv2D:output:00rcnn_1/conv2d_202/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_202/BiasAdd
rcnn_1/conv2d_202/ReluRelu"rcnn_1/conv2d_202/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_202/ReluÒ
-rcnn_1/batch_normalization_196/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_196_readvariableop_resource*
_output_shapes	
:*
dtype02/
-rcnn_1/batch_normalization_196/ReadVariableOpØ
/rcnn_1/batch_normalization_196/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_196_readvariableop_1_resource*
_output_shapes	
:*
dtype021
/rcnn_1/batch_normalization_196/ReadVariableOp_1
>rcnn_1/batch_normalization_196/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_196_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02@
>rcnn_1/batch_normalization_196/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_196/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_196_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02B
@rcnn_1/batch_normalization_196/FusedBatchNormV3/ReadVariableOp_1§
/rcnn_1/batch_normalization_196/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_202/Relu:activations:05rcnn_1/batch_normalization_196/ReadVariableOp:value:07rcnn_1/batch_normalization_196/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_196/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_196/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_196/FusedBatchNormV3Ì
'rcnn_1/conv2d_203/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_203_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02)
'rcnn_1/conv2d_203/Conv2D/ReadVariableOp
rcnn_1/conv2d_203/Conv2DConv2D3rcnn_1/batch_normalization_196/FusedBatchNormV3:y:0/rcnn_1/conv2d_203/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
rcnn_1/conv2d_203/Conv2DÂ
(rcnn_1/conv2d_203/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_203_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(rcnn_1/conv2d_203/BiasAdd/ReadVariableOpÐ
rcnn_1/conv2d_203/BiasAddBiasAdd!rcnn_1/conv2d_203/Conv2D:output:00rcnn_1/conv2d_203/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_203/BiasAdd
rcnn_1/conv2d_203/ReluRelu"rcnn_1/conv2d_203/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_203/ReluÑ
-rcnn_1/batch_normalization_197/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_197_readvariableop_resource*
_output_shapes
:0*
dtype02/
-rcnn_1/batch_normalization_197/ReadVariableOp×
/rcnn_1/batch_normalization_197/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_197_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/rcnn_1/batch_normalization_197/ReadVariableOp_1
>rcnn_1/batch_normalization_197/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_197_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>rcnn_1/batch_normalization_197/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_197/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_197_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@rcnn_1/batch_normalization_197/FusedBatchNormV3/ReadVariableOp_1¢
/rcnn_1/batch_normalization_197/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_203/Relu:activations:05rcnn_1/batch_normalization_197/ReadVariableOp:value:07rcnn_1/batch_normalization_197/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_197/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_197/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_197/FusedBatchNormV3ð
rcnn_1/max_pooling2d_10/MaxPoolMaxPool3rcnn_1/batch_normalization_196/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2!
rcnn_1/max_pooling2d_10/MaxPoolw
rcnn_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rcnn_1/concat_2/axisð
rcnn_1/concat_2ConcatV2(rcnn_1/max_pooling2d_10/MaxPool:output:03rcnn_1/batch_normalization_197/FusedBatchNormV3:y:0rcnn_1/concat_2/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/concat_2
rcnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
rcnn_1/strided_slice_3/stack
rcnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2 
rcnn_1/strided_slice_3/stack_1
rcnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
rcnn_1/strided_slice_3/stack_2Á
rcnn_1/strided_slice_3StridedSlice	input_2_4%rcnn_1/strided_slice_3/stack:output:0'rcnn_1/strided_slice_3/stack_1:output:0'rcnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask2
rcnn_1/strided_slice_3m
rcnn_1/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
rcnn_1/Greater_3/y£
rcnn_1/Greater_3Greaterrcnn_1/strided_slice_3:output:0rcnn_1/Greater_3/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Greater_3h
rcnn_1/Where_3Wherercnn_1/Greater_3:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Where_3n
rcnn_1/map_3/ShapeShapercnn_1/Where_3:index:0*
T0	*
_output_shapes
:2
rcnn_1/map_3/Shape
 rcnn_1/map_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 rcnn_1/map_3/strided_slice/stack
"rcnn_1/map_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"rcnn_1/map_3/strided_slice/stack_1
"rcnn_1/map_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rcnn_1/map_3/strided_slice/stack_2°
rcnn_1/map_3/strided_sliceStridedSlicercnn_1/map_3/Shape:output:0)rcnn_1/map_3/strided_slice/stack:output:0+rcnn_1/map_3/strided_slice/stack_1:output:0+rcnn_1/map_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rcnn_1/map_3/strided_slice
(rcnn_1/map_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(rcnn_1/map_3/TensorArrayV2/element_shapeä
rcnn_1/map_3/TensorArrayV2TensorListReserve1rcnn_1/map_3/TensorArrayV2/element_shape:output:0#rcnn_1/map_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type02
rcnn_1/map_3/TensorArrayV2Ò
Brcnn_1/map_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Brcnn_1/map_3/TensorArrayUnstack/TensorListFromTensor/element_shape¨
4rcnn_1/map_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrcnn_1/Where_3:index:0Krcnn_1/map_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0	*

shape_type026
4rcnn_1/map_3/TensorArrayUnstack/TensorListFromTensorj
rcnn_1/map_3/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
rcnn_1/map_3/Const­
*rcnn_1/map_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2,
*rcnn_1/map_3/TensorArrayV2_1/element_shapeê
rcnn_1/map_3/TensorArrayV2_1TensorListReserve3rcnn_1/map_3/TensorArrayV2_1/element_shape:output:0#rcnn_1/map_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rcnn_1/map_3/TensorArrayV2_1
rcnn_1/map_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
rcnn_1/map_3/while/loop_counterÞ
rcnn_1/map_3/whileStatelessWhile(rcnn_1/map_3/while/loop_counter:output:0#rcnn_1/map_3/strided_slice:output:0rcnn_1/map_3/Const:output:0%rcnn_1/map_3/TensorArrayV2_1:handle:0#rcnn_1/map_3/strided_slice:output:0Drcnn_1/map_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0rcnn_1_map_while_input_6rcnn_1_map_while_input_7rcnn_1/Pad:output:0*
T
2			*
_lower_using_switch_merge(*
_num_original_outputs	*H
_output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *)
body!R
rcnn_1_map_3_while_body_53399*)
cond!R
rcnn_1_map_3_while_cond_53398*G
output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/map_3/whileÓ
=rcnn_1/map_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2?
=rcnn_1/map_3/TensorArrayV2Stack/TensorListStack/element_shape¡
/rcnn_1/map_3/TensorArrayV2Stack/TensorListStackTensorListStackrcnn_1/map_3/while:output:3Frcnn_1/map_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
element_dtype021
/rcnn_1/map_3/TensorArrayV2Stack/TensorListStackÍ
'rcnn_1/conv2d_200/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_200_conv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02)
'rcnn_1/conv2d_200/Conv2D/ReadVariableOp
rcnn_1/conv2d_200/Conv2DConv2D8rcnn_1/map_3/TensorArrayV2Stack/TensorListStack:tensor:0/rcnn_1/conv2d_200/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
rcnn_1/conv2d_200/Conv2DÃ
(rcnn_1/conv2d_200/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_200_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/conv2d_200/BiasAdd/ReadVariableOpÑ
rcnn_1/conv2d_200/BiasAddBiasAdd!rcnn_1/conv2d_200/Conv2D:output:00rcnn_1/conv2d_200/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_200/BiasAdd
rcnn_1/conv2d_200/ReluRelu"rcnn_1/conv2d_200/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_200/ReluÒ
-rcnn_1/batch_normalization_194/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_194_readvariableop_resource*
_output_shapes	
:*
dtype02/
-rcnn_1/batch_normalization_194/ReadVariableOpØ
/rcnn_1/batch_normalization_194/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_194_readvariableop_1_resource*
_output_shapes	
:*
dtype021
/rcnn_1/batch_normalization_194/ReadVariableOp_1
>rcnn_1/batch_normalization_194/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_194_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02@
>rcnn_1/batch_normalization_194/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_194/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_194_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02B
@rcnn_1/batch_normalization_194/FusedBatchNormV3/ReadVariableOp_1§
/rcnn_1/batch_normalization_194/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_200/Relu:activations:05rcnn_1/batch_normalization_194/ReadVariableOp:value:07rcnn_1/batch_normalization_194/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_194/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_194/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_194/FusedBatchNormV3Ì
'rcnn_1/conv2d_201/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_201_conv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02)
'rcnn_1/conv2d_201/Conv2D/ReadVariableOp
rcnn_1/conv2d_201/Conv2DConv2D3rcnn_1/batch_normalization_194/FusedBatchNormV3:y:0/rcnn_1/conv2d_201/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
rcnn_1/conv2d_201/Conv2DÂ
(rcnn_1/conv2d_201/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_201_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(rcnn_1/conv2d_201/BiasAdd/ReadVariableOpÐ
rcnn_1/conv2d_201/BiasAddBiasAdd!rcnn_1/conv2d_201/Conv2D:output:00rcnn_1/conv2d_201/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_201/BiasAdd
rcnn_1/conv2d_201/ReluRelu"rcnn_1/conv2d_201/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_201/ReluÑ
-rcnn_1/batch_normalization_195/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_195_readvariableop_resource*
_output_shapes
:0*
dtype02/
-rcnn_1/batch_normalization_195/ReadVariableOp×
/rcnn_1/batch_normalization_195/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_195_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/rcnn_1/batch_normalization_195/ReadVariableOp_1
>rcnn_1/batch_normalization_195/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_195_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>rcnn_1/batch_normalization_195/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_195_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@rcnn_1/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1¢
/rcnn_1/batch_normalization_195/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_201/Relu:activations:05rcnn_1/batch_normalization_195/ReadVariableOp:value:07rcnn_1/batch_normalization_195/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_195/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_195/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_195/FusedBatchNormV3î
rcnn_1/max_pooling2d_9/MaxPoolMaxPool3rcnn_1/batch_normalization_194/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2 
rcnn_1/max_pooling2d_9/MaxPoolw
rcnn_1/concat_3/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rcnn_1/concat_3/axisï
rcnn_1/concat_3ConcatV2'rcnn_1/max_pooling2d_9/MaxPool:output:03rcnn_1/batch_normalization_195/FusedBatchNormV3:y:0rcnn_1/concat_3/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/concat_3
rcnn_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
rcnn_1/strided_slice_4/stack
rcnn_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2 
rcnn_1/strided_slice_4/stack_1
rcnn_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
rcnn_1/strided_slice_4/stack_2Á
rcnn_1/strided_slice_4StridedSlice	input_2_5%rcnn_1/strided_slice_4/stack:output:0'rcnn_1/strided_slice_4/stack_1:output:0'rcnn_1/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask2
rcnn_1/strided_slice_4m
rcnn_1/Greater_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
rcnn_1/Greater_4/y£
rcnn_1/Greater_4Greaterrcnn_1/strided_slice_4:output:0rcnn_1/Greater_4/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Greater_4h
rcnn_1/Where_4Wherercnn_1/Greater_4:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/Where_4n
rcnn_1/map_4/ShapeShapercnn_1/Where_4:index:0*
T0	*
_output_shapes
:2
rcnn_1/map_4/Shape
 rcnn_1/map_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 rcnn_1/map_4/strided_slice/stack
"rcnn_1/map_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"rcnn_1/map_4/strided_slice/stack_1
"rcnn_1/map_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rcnn_1/map_4/strided_slice/stack_2°
rcnn_1/map_4/strided_sliceStridedSlicercnn_1/map_4/Shape:output:0)rcnn_1/map_4/strided_slice/stack:output:0+rcnn_1/map_4/strided_slice/stack_1:output:0+rcnn_1/map_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rcnn_1/map_4/strided_slice
(rcnn_1/map_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(rcnn_1/map_4/TensorArrayV2/element_shapeä
rcnn_1/map_4/TensorArrayV2TensorListReserve1rcnn_1/map_4/TensorArrayV2/element_shape:output:0#rcnn_1/map_4/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type02
rcnn_1/map_4/TensorArrayV2Ò
Brcnn_1/map_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Brcnn_1/map_4/TensorArrayUnstack/TensorListFromTensor/element_shape¨
4rcnn_1/map_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrcnn_1/Where_4:index:0Krcnn_1/map_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0	*

shape_type026
4rcnn_1/map_4/TensorArrayUnstack/TensorListFromTensorj
rcnn_1/map_4/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
rcnn_1/map_4/Const­
*rcnn_1/map_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2,
*rcnn_1/map_4/TensorArrayV2_1/element_shapeê
rcnn_1/map_4/TensorArrayV2_1TensorListReserve3rcnn_1/map_4/TensorArrayV2_1/element_shape:output:0#rcnn_1/map_4/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rcnn_1/map_4/TensorArrayV2_1
rcnn_1/map_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
rcnn_1/map_4/while/loop_counterÞ
rcnn_1/map_4/whileStatelessWhile(rcnn_1/map_4/while/loop_counter:output:0#rcnn_1/map_4/strided_slice:output:0rcnn_1/map_4/Const:output:0%rcnn_1/map_4/TensorArrayV2_1:handle:0#rcnn_1/map_4/strided_slice:output:0Drcnn_1/map_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0rcnn_1_map_while_input_6rcnn_1_map_while_input_7rcnn_1/Pad:output:0*
T
2			*
_lower_using_switch_merge(*
_num_original_outputs	*H
_output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *)
body!R
rcnn_1_map_4_while_body_53527*)
cond!R
rcnn_1_map_4_while_cond_53526*G
output_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/map_4/whileÓ
=rcnn_1/map_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"      À   2?
=rcnn_1/map_4/TensorArrayV2Stack/TensorListStack/element_shape¡
/rcnn_1/map_4/TensorArrayV2Stack/TensorListStackTensorListStackrcnn_1/map_4/while:output:3Frcnn_1/map_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
element_dtype021
/rcnn_1/map_4/TensorArrayV2Stack/TensorListStackÍ
'rcnn_1/conv2d_198/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_198_conv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02)
'rcnn_1/conv2d_198/Conv2D/ReadVariableOp
rcnn_1/conv2d_198/Conv2DConv2D8rcnn_1/map_4/TensorArrayV2Stack/TensorListStack:tensor:0/rcnn_1/conv2d_198/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
rcnn_1/conv2d_198/Conv2DÃ
(rcnn_1/conv2d_198/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_198_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/conv2d_198/BiasAdd/ReadVariableOpÑ
rcnn_1/conv2d_198/BiasAddBiasAdd!rcnn_1/conv2d_198/Conv2D:output:00rcnn_1/conv2d_198/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_198/BiasAdd
rcnn_1/conv2d_198/ReluRelu"rcnn_1/conv2d_198/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/conv2d_198/ReluÒ
-rcnn_1/batch_normalization_192/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_192_readvariableop_resource*
_output_shapes	
:*
dtype02/
-rcnn_1/batch_normalization_192/ReadVariableOpØ
/rcnn_1/batch_normalization_192/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_192_readvariableop_1_resource*
_output_shapes	
:*
dtype021
/rcnn_1/batch_normalization_192/ReadVariableOp_1
>rcnn_1/batch_normalization_192/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_192_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02@
>rcnn_1/batch_normalization_192/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_192/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_192_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02B
@rcnn_1/batch_normalization_192/FusedBatchNormV3/ReadVariableOp_1§
/rcnn_1/batch_normalization_192/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_198/Relu:activations:05rcnn_1/batch_normalization_192/ReadVariableOp:value:07rcnn_1/batch_normalization_192/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_192/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_192/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_192/FusedBatchNormV3Ì
'rcnn_1/conv2d_199/Conv2D/ReadVariableOpReadVariableOp0rcnn_1_conv2d_199_conv2d_readvariableop_resource*'
_output_shapes
:		0*
dtype02)
'rcnn_1/conv2d_199/Conv2D/ReadVariableOp
rcnn_1/conv2d_199/Conv2DConv2D3rcnn_1/batch_normalization_192/FusedBatchNormV3:y:0/rcnn_1/conv2d_199/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
rcnn_1/conv2d_199/Conv2DÂ
(rcnn_1/conv2d_199/BiasAdd/ReadVariableOpReadVariableOp1rcnn_1_conv2d_199_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02*
(rcnn_1/conv2d_199/BiasAdd/ReadVariableOpÐ
rcnn_1/conv2d_199/BiasAddBiasAdd!rcnn_1/conv2d_199/Conv2D:output:00rcnn_1/conv2d_199/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_199/BiasAdd
rcnn_1/conv2d_199/ReluRelu"rcnn_1/conv2d_199/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
rcnn_1/conv2d_199/ReluÑ
-rcnn_1/batch_normalization_193/ReadVariableOpReadVariableOp6rcnn_1_batch_normalization_193_readvariableop_resource*
_output_shapes
:0*
dtype02/
-rcnn_1/batch_normalization_193/ReadVariableOp×
/rcnn_1/batch_normalization_193/ReadVariableOp_1ReadVariableOp8rcnn_1_batch_normalization_193_readvariableop_1_resource*
_output_shapes
:0*
dtype021
/rcnn_1/batch_normalization_193/ReadVariableOp_1
>rcnn_1/batch_normalization_193/FusedBatchNormV3/ReadVariableOpReadVariableOpGrcnn_1_batch_normalization_193_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02@
>rcnn_1/batch_normalization_193/FusedBatchNormV3/ReadVariableOp
@rcnn_1/batch_normalization_193/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIrcnn_1_batch_normalization_193_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02B
@rcnn_1/batch_normalization_193/FusedBatchNormV3/ReadVariableOp_1¢
/rcnn_1/batch_normalization_193/FusedBatchNormV3FusedBatchNormV3$rcnn_1/conv2d_199/Relu:activations:05rcnn_1/batch_normalization_193/ReadVariableOp:value:07rcnn_1/batch_normalization_193/ReadVariableOp_1:value:0Frcnn_1/batch_normalization_193/FusedBatchNormV3/ReadVariableOp:value:0Hrcnn_1/batch_normalization_193/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 21
/rcnn_1/batch_normalization_193/FusedBatchNormV3î
rcnn_1/max_pooling2d_8/MaxPoolMaxPool3rcnn_1/batch_normalization_192/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
		*
paddingVALID*
strides
2 
rcnn_1/max_pooling2d_8/MaxPoolw
rcnn_1/concat_4/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rcnn_1/concat_4/axisï
rcnn_1/concat_4ConcatV2'rcnn_1/max_pooling2d_8/MaxPool:output:03rcnn_1/batch_normalization_193/FusedBatchNormV3:y:0rcnn_1/concat_4/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/concat_4
rcnn_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  2
rcnn_1/flatten_1/Const«
rcnn_1/flatten_1/ReshapeReshapercnn_1/concat:output:0rcnn_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/flatten_1/Reshape
rcnn_1/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  2
rcnn_1/flatten_1/Const_1³
rcnn_1/flatten_1/Reshape_1Reshapercnn_1/concat_1:output:0!rcnn_1/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/flatten_1/Reshape_1
rcnn_1/flatten_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  2
rcnn_1/flatten_1/Const_2³
rcnn_1/flatten_1/Reshape_2Reshapercnn_1/concat_2:output:0!rcnn_1/flatten_1/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/flatten_1/Reshape_2
rcnn_1/flatten_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  2
rcnn_1/flatten_1/Const_3³
rcnn_1/flatten_1/Reshape_3Reshapercnn_1/concat_3:output:0!rcnn_1/flatten_1/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/flatten_1/Reshape_3
rcnn_1/flatten_1/Const_4Const*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  2
rcnn_1/flatten_1/Const_4³
rcnn_1/flatten_1/Reshape_4Reshapercnn_1/concat_4:output:0!rcnn_1/flatten_1/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
rcnn_1/flatten_1/Reshape_4¼
$rcnn_1/dense_6/MatMul/ReadVariableOpReadVariableOp-rcnn_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02&
$rcnn_1/dense_6/MatMul/ReadVariableOp¼
rcnn_1/dense_6/MatMulMatMul!rcnn_1/flatten_1/Reshape:output:0,rcnn_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/MatMulº
%rcnn_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp.rcnn_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%rcnn_1/dense_6/BiasAdd/ReadVariableOp¾
rcnn_1/dense_6/BiasAddBiasAddrcnn_1/dense_6/MatMul:product:0-rcnn_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/BiasAdd
rcnn_1/dense_6/ReluRelurcnn_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/ReluÀ
&rcnn_1/dense_6/MatMul_1/ReadVariableOpReadVariableOp-rcnn_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02(
&rcnn_1/dense_6/MatMul_1/ReadVariableOpÄ
rcnn_1/dense_6/MatMul_1MatMul#rcnn_1/flatten_1/Reshape_1:output:0.rcnn_1/dense_6/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/MatMul_1¾
'rcnn_1/dense_6/BiasAdd_1/ReadVariableOpReadVariableOp.rcnn_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'rcnn_1/dense_6/BiasAdd_1/ReadVariableOpÆ
rcnn_1/dense_6/BiasAdd_1BiasAdd!rcnn_1/dense_6/MatMul_1:product:0/rcnn_1/dense_6/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/BiasAdd_1
rcnn_1/dense_6/Relu_1Relu!rcnn_1/dense_6/BiasAdd_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/Relu_1À
&rcnn_1/dense_6/MatMul_2/ReadVariableOpReadVariableOp-rcnn_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02(
&rcnn_1/dense_6/MatMul_2/ReadVariableOpÄ
rcnn_1/dense_6/MatMul_2MatMul#rcnn_1/flatten_1/Reshape_2:output:0.rcnn_1/dense_6/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/MatMul_2¾
'rcnn_1/dense_6/BiasAdd_2/ReadVariableOpReadVariableOp.rcnn_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'rcnn_1/dense_6/BiasAdd_2/ReadVariableOpÆ
rcnn_1/dense_6/BiasAdd_2BiasAdd!rcnn_1/dense_6/MatMul_2:product:0/rcnn_1/dense_6/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/BiasAdd_2
rcnn_1/dense_6/Relu_2Relu!rcnn_1/dense_6/BiasAdd_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/Relu_2À
&rcnn_1/dense_6/MatMul_3/ReadVariableOpReadVariableOp-rcnn_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02(
&rcnn_1/dense_6/MatMul_3/ReadVariableOpÄ
rcnn_1/dense_6/MatMul_3MatMul#rcnn_1/flatten_1/Reshape_3:output:0.rcnn_1/dense_6/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/MatMul_3¾
'rcnn_1/dense_6/BiasAdd_3/ReadVariableOpReadVariableOp.rcnn_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'rcnn_1/dense_6/BiasAdd_3/ReadVariableOpÆ
rcnn_1/dense_6/BiasAdd_3BiasAdd!rcnn_1/dense_6/MatMul_3:product:0/rcnn_1/dense_6/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/BiasAdd_3
rcnn_1/dense_6/Relu_3Relu!rcnn_1/dense_6/BiasAdd_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/Relu_3À
&rcnn_1/dense_6/MatMul_4/ReadVariableOpReadVariableOp-rcnn_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02(
&rcnn_1/dense_6/MatMul_4/ReadVariableOpÄ
rcnn_1/dense_6/MatMul_4MatMul#rcnn_1/flatten_1/Reshape_4:output:0.rcnn_1/dense_6/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/MatMul_4¾
'rcnn_1/dense_6/BiasAdd_4/ReadVariableOpReadVariableOp.rcnn_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'rcnn_1/dense_6/BiasAdd_4/ReadVariableOpÆ
rcnn_1/dense_6/BiasAdd_4BiasAdd!rcnn_1/dense_6/MatMul_4:product:0/rcnn_1/dense_6/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/BiasAdd_4
rcnn_1/dense_6/Relu_4Relu!rcnn_1/dense_6/BiasAdd_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_6/Relu_4»
$rcnn_1/dense_7/MatMul/ReadVariableOpReadVariableOp-rcnn_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02&
$rcnn_1/dense_7/MatMul/ReadVariableOp»
rcnn_1/dense_7/MatMulMatMul!rcnn_1/dense_6/Relu:activations:0,rcnn_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/MatMul¹
%rcnn_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp.rcnn_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%rcnn_1/dense_7/BiasAdd/ReadVariableOp½
rcnn_1/dense_7/BiasAddBiasAddrcnn_1/dense_7/MatMul:product:0-rcnn_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/BiasAdd
rcnn_1/dense_7/ReluRelurcnn_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/Relu¿
&rcnn_1/dense_7/MatMul_1/ReadVariableOpReadVariableOp-rcnn_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&rcnn_1/dense_7/MatMul_1/ReadVariableOpÃ
rcnn_1/dense_7/MatMul_1MatMul#rcnn_1/dense_6/Relu_1:activations:0.rcnn_1/dense_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/MatMul_1½
'rcnn_1/dense_7/BiasAdd_1/ReadVariableOpReadVariableOp.rcnn_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'rcnn_1/dense_7/BiasAdd_1/ReadVariableOpÅ
rcnn_1/dense_7/BiasAdd_1BiasAdd!rcnn_1/dense_7/MatMul_1:product:0/rcnn_1/dense_7/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/BiasAdd_1
rcnn_1/dense_7/Relu_1Relu!rcnn_1/dense_7/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/Relu_1¿
&rcnn_1/dense_7/MatMul_2/ReadVariableOpReadVariableOp-rcnn_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&rcnn_1/dense_7/MatMul_2/ReadVariableOpÃ
rcnn_1/dense_7/MatMul_2MatMul#rcnn_1/dense_6/Relu_2:activations:0.rcnn_1/dense_7/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/MatMul_2½
'rcnn_1/dense_7/BiasAdd_2/ReadVariableOpReadVariableOp.rcnn_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'rcnn_1/dense_7/BiasAdd_2/ReadVariableOpÅ
rcnn_1/dense_7/BiasAdd_2BiasAdd!rcnn_1/dense_7/MatMul_2:product:0/rcnn_1/dense_7/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/BiasAdd_2
rcnn_1/dense_7/Relu_2Relu!rcnn_1/dense_7/BiasAdd_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/Relu_2¿
&rcnn_1/dense_7/MatMul_3/ReadVariableOpReadVariableOp-rcnn_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&rcnn_1/dense_7/MatMul_3/ReadVariableOpÃ
rcnn_1/dense_7/MatMul_3MatMul#rcnn_1/dense_6/Relu_3:activations:0.rcnn_1/dense_7/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/MatMul_3½
'rcnn_1/dense_7/BiasAdd_3/ReadVariableOpReadVariableOp.rcnn_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'rcnn_1/dense_7/BiasAdd_3/ReadVariableOpÅ
rcnn_1/dense_7/BiasAdd_3BiasAdd!rcnn_1/dense_7/MatMul_3:product:0/rcnn_1/dense_7/BiasAdd_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/BiasAdd_3
rcnn_1/dense_7/Relu_3Relu!rcnn_1/dense_7/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/Relu_3¿
&rcnn_1/dense_7/MatMul_4/ReadVariableOpReadVariableOp-rcnn_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&rcnn_1/dense_7/MatMul_4/ReadVariableOpÃ
rcnn_1/dense_7/MatMul_4MatMul#rcnn_1/dense_6/Relu_4:activations:0.rcnn_1/dense_7/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/MatMul_4½
'rcnn_1/dense_7/BiasAdd_4/ReadVariableOpReadVariableOp.rcnn_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'rcnn_1/dense_7/BiasAdd_4/ReadVariableOpÅ
rcnn_1/dense_7/BiasAdd_4BiasAdd!rcnn_1/dense_7/MatMul_4:product:0/rcnn_1/dense_7/BiasAdd_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/BiasAdd_4
rcnn_1/dense_7/Relu_4Relu!rcnn_1/dense_7/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
rcnn_1/dense_7/Relu_4º
$rcnn_1/dense_8/MatMul/ReadVariableOpReadVariableOp-rcnn_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$rcnn_1/dense_8/MatMul/ReadVariableOp»
rcnn_1/dense_8/MatMulMatMul!rcnn_1/dense_7/Relu:activations:0,rcnn_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/MatMul¹
%rcnn_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp.rcnn_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%rcnn_1/dense_8/BiasAdd/ReadVariableOp½
rcnn_1/dense_8/BiasAddBiasAddrcnn_1/dense_8/MatMul:product:0-rcnn_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/BiasAdd
rcnn_1/dense_8/SoftmaxSoftmaxrcnn_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/Softmax¾
&rcnn_1/dense_8/MatMul_1/ReadVariableOpReadVariableOp-rcnn_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&rcnn_1/dense_8/MatMul_1/ReadVariableOpÃ
rcnn_1/dense_8/MatMul_1MatMul#rcnn_1/dense_7/Relu_1:activations:0.rcnn_1/dense_8/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/MatMul_1½
'rcnn_1/dense_8/BiasAdd_1/ReadVariableOpReadVariableOp.rcnn_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rcnn_1/dense_8/BiasAdd_1/ReadVariableOpÅ
rcnn_1/dense_8/BiasAdd_1BiasAdd!rcnn_1/dense_8/MatMul_1:product:0/rcnn_1/dense_8/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/BiasAdd_1
rcnn_1/dense_8/Softmax_1Softmax!rcnn_1/dense_8/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/Softmax_1¾
&rcnn_1/dense_8/MatMul_2/ReadVariableOpReadVariableOp-rcnn_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&rcnn_1/dense_8/MatMul_2/ReadVariableOpÃ
rcnn_1/dense_8/MatMul_2MatMul#rcnn_1/dense_7/Relu_2:activations:0.rcnn_1/dense_8/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/MatMul_2½
'rcnn_1/dense_8/BiasAdd_2/ReadVariableOpReadVariableOp.rcnn_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rcnn_1/dense_8/BiasAdd_2/ReadVariableOpÅ
rcnn_1/dense_8/BiasAdd_2BiasAdd!rcnn_1/dense_8/MatMul_2:product:0/rcnn_1/dense_8/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/BiasAdd_2
rcnn_1/dense_8/Softmax_2Softmax!rcnn_1/dense_8/BiasAdd_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/Softmax_2¾
&rcnn_1/dense_8/MatMul_3/ReadVariableOpReadVariableOp-rcnn_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&rcnn_1/dense_8/MatMul_3/ReadVariableOpÃ
rcnn_1/dense_8/MatMul_3MatMul#rcnn_1/dense_7/Relu_3:activations:0.rcnn_1/dense_8/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/MatMul_3½
'rcnn_1/dense_8/BiasAdd_3/ReadVariableOpReadVariableOp.rcnn_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rcnn_1/dense_8/BiasAdd_3/ReadVariableOpÅ
rcnn_1/dense_8/BiasAdd_3BiasAdd!rcnn_1/dense_8/MatMul_3:product:0/rcnn_1/dense_8/BiasAdd_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/BiasAdd_3
rcnn_1/dense_8/Softmax_3Softmax!rcnn_1/dense_8/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/Softmax_3¾
&rcnn_1/dense_8/MatMul_4/ReadVariableOpReadVariableOp-rcnn_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&rcnn_1/dense_8/MatMul_4/ReadVariableOpÃ
rcnn_1/dense_8/MatMul_4MatMul#rcnn_1/dense_7/Relu_4:activations:0.rcnn_1/dense_8/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/MatMul_4½
'rcnn_1/dense_8/BiasAdd_4/ReadVariableOpReadVariableOp.rcnn_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rcnn_1/dense_8/BiasAdd_4/ReadVariableOpÅ
rcnn_1/dense_8/BiasAdd_4BiasAdd!rcnn_1/dense_8/MatMul_4:product:0/rcnn_1/dense_8/BiasAdd_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/BiasAdd_4
rcnn_1/dense_8/Softmax_4Softmax!rcnn_1/dense_8/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_8/Softmax_4¼
$rcnn_1/dense_9/MatMul/ReadVariableOpReadVariableOp-rcnn_1_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02&
$rcnn_1/dense_9/MatMul/ReadVariableOp¼
rcnn_1/dense_9/MatMulMatMul!rcnn_1/flatten_1/Reshape:output:0,rcnn_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/MatMulº
%rcnn_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp.rcnn_1_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%rcnn_1/dense_9/BiasAdd/ReadVariableOp¾
rcnn_1/dense_9/BiasAddBiasAddrcnn_1/dense_9/MatMul:product:0-rcnn_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/BiasAdd
rcnn_1/dense_9/ReluRelurcnn_1/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/ReluÀ
&rcnn_1/dense_9/MatMul_1/ReadVariableOpReadVariableOp-rcnn_1_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02(
&rcnn_1/dense_9/MatMul_1/ReadVariableOpÄ
rcnn_1/dense_9/MatMul_1MatMul#rcnn_1/flatten_1/Reshape_1:output:0.rcnn_1/dense_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/MatMul_1¾
'rcnn_1/dense_9/BiasAdd_1/ReadVariableOpReadVariableOp.rcnn_1_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'rcnn_1/dense_9/BiasAdd_1/ReadVariableOpÆ
rcnn_1/dense_9/BiasAdd_1BiasAdd!rcnn_1/dense_9/MatMul_1:product:0/rcnn_1/dense_9/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/BiasAdd_1
rcnn_1/dense_9/Relu_1Relu!rcnn_1/dense_9/BiasAdd_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/Relu_1À
&rcnn_1/dense_9/MatMul_2/ReadVariableOpReadVariableOp-rcnn_1_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02(
&rcnn_1/dense_9/MatMul_2/ReadVariableOpÄ
rcnn_1/dense_9/MatMul_2MatMul#rcnn_1/flatten_1/Reshape_2:output:0.rcnn_1/dense_9/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/MatMul_2¾
'rcnn_1/dense_9/BiasAdd_2/ReadVariableOpReadVariableOp.rcnn_1_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'rcnn_1/dense_9/BiasAdd_2/ReadVariableOpÆ
rcnn_1/dense_9/BiasAdd_2BiasAdd!rcnn_1/dense_9/MatMul_2:product:0/rcnn_1/dense_9/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/BiasAdd_2
rcnn_1/dense_9/Relu_2Relu!rcnn_1/dense_9/BiasAdd_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/Relu_2À
&rcnn_1/dense_9/MatMul_3/ReadVariableOpReadVariableOp-rcnn_1_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02(
&rcnn_1/dense_9/MatMul_3/ReadVariableOpÄ
rcnn_1/dense_9/MatMul_3MatMul#rcnn_1/flatten_1/Reshape_3:output:0.rcnn_1/dense_9/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/MatMul_3¾
'rcnn_1/dense_9/BiasAdd_3/ReadVariableOpReadVariableOp.rcnn_1_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'rcnn_1/dense_9/BiasAdd_3/ReadVariableOpÆ
rcnn_1/dense_9/BiasAdd_3BiasAdd!rcnn_1/dense_9/MatMul_3:product:0/rcnn_1/dense_9/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/BiasAdd_3
rcnn_1/dense_9/Relu_3Relu!rcnn_1/dense_9/BiasAdd_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/Relu_3À
&rcnn_1/dense_9/MatMul_4/ReadVariableOpReadVariableOp-rcnn_1_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02(
&rcnn_1/dense_9/MatMul_4/ReadVariableOpÄ
rcnn_1/dense_9/MatMul_4MatMul#rcnn_1/flatten_1/Reshape_4:output:0.rcnn_1/dense_9/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/MatMul_4¾
'rcnn_1/dense_9/BiasAdd_4/ReadVariableOpReadVariableOp.rcnn_1_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'rcnn_1/dense_9/BiasAdd_4/ReadVariableOpÆ
rcnn_1/dense_9/BiasAdd_4BiasAdd!rcnn_1/dense_9/MatMul_4:product:0/rcnn_1/dense_9/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/BiasAdd_4
rcnn_1/dense_9/Relu_4Relu!rcnn_1/dense_9/BiasAdd_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_9/Relu_4¿
%rcnn_1/dense_10/MatMul/ReadVariableOpReadVariableOp.rcnn_1_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02'
%rcnn_1/dense_10/MatMul/ReadVariableOp¿
rcnn_1/dense_10/MatMulMatMul!rcnn_1/dense_9/Relu:activations:0-rcnn_1/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/MatMul½
&rcnn_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp/rcnn_1_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02(
&rcnn_1/dense_10/BiasAdd/ReadVariableOpÂ
rcnn_1/dense_10/BiasAddBiasAdd rcnn_1/dense_10/MatMul:product:0.rcnn_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/BiasAdd
rcnn_1/dense_10/ReluRelu rcnn_1/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/ReluÃ
'rcnn_1/dense_10/MatMul_1/ReadVariableOpReadVariableOp.rcnn_1_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'rcnn_1/dense_10/MatMul_1/ReadVariableOpÇ
rcnn_1/dense_10/MatMul_1MatMul#rcnn_1/dense_9/Relu_1:activations:0/rcnn_1/dense_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/MatMul_1Á
(rcnn_1/dense_10/BiasAdd_1/ReadVariableOpReadVariableOp/rcnn_1_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/dense_10/BiasAdd_1/ReadVariableOpÊ
rcnn_1/dense_10/BiasAdd_1BiasAdd"rcnn_1/dense_10/MatMul_1:product:00rcnn_1/dense_10/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/BiasAdd_1
rcnn_1/dense_10/Relu_1Relu"rcnn_1/dense_10/BiasAdd_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/Relu_1Ã
'rcnn_1/dense_10/MatMul_2/ReadVariableOpReadVariableOp.rcnn_1_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'rcnn_1/dense_10/MatMul_2/ReadVariableOpÇ
rcnn_1/dense_10/MatMul_2MatMul#rcnn_1/dense_9/Relu_2:activations:0/rcnn_1/dense_10/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/MatMul_2Á
(rcnn_1/dense_10/BiasAdd_2/ReadVariableOpReadVariableOp/rcnn_1_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/dense_10/BiasAdd_2/ReadVariableOpÊ
rcnn_1/dense_10/BiasAdd_2BiasAdd"rcnn_1/dense_10/MatMul_2:product:00rcnn_1/dense_10/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/BiasAdd_2
rcnn_1/dense_10/Relu_2Relu"rcnn_1/dense_10/BiasAdd_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/Relu_2Ã
'rcnn_1/dense_10/MatMul_3/ReadVariableOpReadVariableOp.rcnn_1_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'rcnn_1/dense_10/MatMul_3/ReadVariableOpÇ
rcnn_1/dense_10/MatMul_3MatMul#rcnn_1/dense_9/Relu_3:activations:0/rcnn_1/dense_10/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/MatMul_3Á
(rcnn_1/dense_10/BiasAdd_3/ReadVariableOpReadVariableOp/rcnn_1_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/dense_10/BiasAdd_3/ReadVariableOpÊ
rcnn_1/dense_10/BiasAdd_3BiasAdd"rcnn_1/dense_10/MatMul_3:product:00rcnn_1/dense_10/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/BiasAdd_3
rcnn_1/dense_10/Relu_3Relu"rcnn_1/dense_10/BiasAdd_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/Relu_3Ã
'rcnn_1/dense_10/MatMul_4/ReadVariableOpReadVariableOp.rcnn_1_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'rcnn_1/dense_10/MatMul_4/ReadVariableOpÇ
rcnn_1/dense_10/MatMul_4MatMul#rcnn_1/dense_9/Relu_4:activations:0/rcnn_1/dense_10/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/MatMul_4Á
(rcnn_1/dense_10/BiasAdd_4/ReadVariableOpReadVariableOp/rcnn_1_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(rcnn_1/dense_10/BiasAdd_4/ReadVariableOpÊ
rcnn_1/dense_10/BiasAdd_4BiasAdd"rcnn_1/dense_10/MatMul_4:product:00rcnn_1/dense_10/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/BiasAdd_4
rcnn_1/dense_10/Relu_4Relu"rcnn_1/dense_10/BiasAdd_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_10/Relu_4¾
%rcnn_1/dense_11/MatMul/ReadVariableOpReadVariableOp.rcnn_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02'
%rcnn_1/dense_11/MatMul/ReadVariableOp¿
rcnn_1/dense_11/MatMulMatMul"rcnn_1/dense_10/Relu:activations:0-rcnn_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/MatMul¼
&rcnn_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp/rcnn_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&rcnn_1/dense_11/BiasAdd/ReadVariableOpÁ
rcnn_1/dense_11/BiasAddBiasAdd rcnn_1/dense_11/MatMul:product:0.rcnn_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/BiasAddÂ
'rcnn_1/dense_11/MatMul_1/ReadVariableOpReadVariableOp.rcnn_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02)
'rcnn_1/dense_11/MatMul_1/ReadVariableOpÇ
rcnn_1/dense_11/MatMul_1MatMul$rcnn_1/dense_10/Relu_1:activations:0/rcnn_1/dense_11/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/MatMul_1À
(rcnn_1/dense_11/BiasAdd_1/ReadVariableOpReadVariableOp/rcnn_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(rcnn_1/dense_11/BiasAdd_1/ReadVariableOpÉ
rcnn_1/dense_11/BiasAdd_1BiasAdd"rcnn_1/dense_11/MatMul_1:product:00rcnn_1/dense_11/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/BiasAdd_1Â
'rcnn_1/dense_11/MatMul_2/ReadVariableOpReadVariableOp.rcnn_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02)
'rcnn_1/dense_11/MatMul_2/ReadVariableOpÇ
rcnn_1/dense_11/MatMul_2MatMul$rcnn_1/dense_10/Relu_2:activations:0/rcnn_1/dense_11/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/MatMul_2À
(rcnn_1/dense_11/BiasAdd_2/ReadVariableOpReadVariableOp/rcnn_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(rcnn_1/dense_11/BiasAdd_2/ReadVariableOpÉ
rcnn_1/dense_11/BiasAdd_2BiasAdd"rcnn_1/dense_11/MatMul_2:product:00rcnn_1/dense_11/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/BiasAdd_2Â
'rcnn_1/dense_11/MatMul_3/ReadVariableOpReadVariableOp.rcnn_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02)
'rcnn_1/dense_11/MatMul_3/ReadVariableOpÇ
rcnn_1/dense_11/MatMul_3MatMul$rcnn_1/dense_10/Relu_3:activations:0/rcnn_1/dense_11/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/MatMul_3À
(rcnn_1/dense_11/BiasAdd_3/ReadVariableOpReadVariableOp/rcnn_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(rcnn_1/dense_11/BiasAdd_3/ReadVariableOpÉ
rcnn_1/dense_11/BiasAdd_3BiasAdd"rcnn_1/dense_11/MatMul_3:product:00rcnn_1/dense_11/BiasAdd_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/BiasAdd_3Â
'rcnn_1/dense_11/MatMul_4/ReadVariableOpReadVariableOp.rcnn_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02)
'rcnn_1/dense_11/MatMul_4/ReadVariableOpÇ
rcnn_1/dense_11/MatMul_4MatMul$rcnn_1/dense_10/Relu_4:activations:0/rcnn_1/dense_11/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/MatMul_4À
(rcnn_1/dense_11/BiasAdd_4/ReadVariableOpReadVariableOp/rcnn_1_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(rcnn_1/dense_11/BiasAdd_4/ReadVariableOpÉ
rcnn_1/dense_11/BiasAdd_4BiasAdd"rcnn_1/dense_11/MatMul_4:product:00rcnn_1/dense_11/BiasAdd_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rcnn_1/dense_11/BiasAdd_4t
IdentityIdentity rcnn_1/dense_8/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityx

Identity_1Identity rcnn_1/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1l

Identity_2Identityrcnn_1/Where:index:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2z

Identity_3Identity"rcnn_1/dense_8/Softmax_1:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3z

Identity_4Identity"rcnn_1/dense_11/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4n

Identity_5Identityrcnn_1/Where_1:index:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_5z

Identity_6Identity"rcnn_1/dense_8/Softmax_2:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_6z

Identity_7Identity"rcnn_1/dense_11/BiasAdd_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_7n

Identity_8Identityrcnn_1/Where_2:index:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_8z

Identity_9Identity"rcnn_1/dense_8/Softmax_3:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_9|
Identity_10Identity"rcnn_1/dense_11/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_10p
Identity_11Identityrcnn_1/Where_3:index:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_11|
Identity_12Identity"rcnn_1/dense_8/Softmax_4:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_12|
Identity_13Identity"rcnn_1/dense_11/BiasAdd_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_13p
Identity_14Identityrcnn_1/Where_4:index:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Identity_14"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*
_input_shapes
ÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_1:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_2:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_3:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_4:ZV
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	input_2_5
¦

R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_54338

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_55444

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_207_layer_call_and_return_conditional_losses_65408

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_56539

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65057

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
}
(__inference_dense_10_layer_call_fn_65656

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_568812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
ª
7__inference_batch_normalization_197_layer_call_fn_64889

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_546622
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý
ª
7__inference_batch_normalization_196_layer_call_fn_64805

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_560092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
ª
7__inference_batch_normalization_190_layer_call_fn_63840

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_538792
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64761

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
ª
7__inference_batch_normalization_199_layer_call_fn_65236

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_558172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64105

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_200_layer_call_fn_64381

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_200_layer_call_and_return_conditional_losses_562302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¾
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_65551

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64567

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ø
h
 __inference_getROIfeature3_53175

inputs	
unknown	
	unknown_0	

rcnn_1_pad
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constß
PartitionedCallPartitionedCallinputsConst:output:0unknown	unknown_0
rcnn_1_pad*
Tin	
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference_getROIfeature_530382
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.::::ÿÿÿÿÿÿÿÿÿÀ:B >

_output_shapes
:
 
_user_specified_nameinputs:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
¯
ª
B__inference_dense_8_layer_call_and_return_conditional_losses_65607

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_54239

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
		*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64779

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_205_layer_call_and_return_conditional_losses_65112

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

é
rcnn_1_map_1_while_cond_531426
2rcnn_1_map_1_while_rcnn_1_map_1_while_loop_counter1
-rcnn_1_map_1_while_rcnn_1_map_1_strided_slice"
rcnn_1_map_1_while_placeholder$
 rcnn_1_map_1_while_placeholder_16
2rcnn_1_map_1_while_less_rcnn_1_map_1_strided_sliceM
Ircnn_1_map_1_while_rcnn_1_map_1_while_cond_53142___redundant_placeholder0M
Ircnn_1_map_1_while_rcnn_1_map_1_while_cond_53142___redundant_placeholder1	M
Ircnn_1_map_1_while_rcnn_1_map_1_while_cond_53142___redundant_placeholder2	M
Ircnn_1_map_1_while_rcnn_1_map_1_while_cond_53142___redundant_placeholder3
rcnn_1_map_1_while_identity
¯
rcnn_1/map_1/while/LessLessrcnn_1_map_1_while_placeholder2rcnn_1_map_1_while_less_rcnn_1_map_1_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_1/while/LessÂ
rcnn_1/map_1/while/Less_1Less2rcnn_1_map_1_while_rcnn_1_map_1_while_loop_counter-rcnn_1_map_1_while_rcnn_1_map_1_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_1/while/Less_1 
rcnn_1/map_1/while/LogicalAnd
LogicalAndrcnn_1/map_1/while/Less_1:z:0rcnn_1/map_1/while/Less:z:0*
_output_shapes
: 2
rcnn_1/map_1/while/LogicalAnd
rcnn_1/map_1/while/IdentityIdentity!rcnn_1/map_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
rcnn_1/map_1/while/Identity"C
rcnn_1_map_1_while_identity$rcnn_1/map_1/while/Identity:output:0*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
	
­
E__inference_conv2d_197_layer_call_and_return_conditional_losses_55233

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
ª
7__inference_batch_normalization_193_layer_call_fn_64297

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_542222
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs


R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_54222

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¥
ª
7__inference_batch_normalization_196_layer_call_fn_64741

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_545582
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_196_layer_call_and_return_conditional_losses_55133

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
«
C__inference_dense_11_layer_call_and_return_conditional_losses_56919

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64465

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_56383

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63873

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó#
ò
rcnn_1_map_3_while_body_533996
2rcnn_1_map_3_while_rcnn_1_map_3_while_loop_counter1
-rcnn_1_map_3_while_rcnn_1_map_3_strided_slice"
rcnn_1_map_3_while_placeholder$
 rcnn_1_map_3_while_placeholder_15
1rcnn_1_map_3_while_rcnn_1_map_3_strided_slice_1_0q
mrcnn_1_map_3_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_3_tensorarrayunstack_tensorlistfromtensor_0
rcnn_1_map_3_while_53432_0	
rcnn_1_map_3_while_53434_0	#
rcnn_1_map_3_while_rcnn_1_pad_0
rcnn_1_map_3_while_identity!
rcnn_1_map_3_while_identity_1!
rcnn_1_map_3_while_identity_2!
rcnn_1_map_3_while_identity_33
/rcnn_1_map_3_while_rcnn_1_map_3_strided_slice_1o
krcnn_1_map_3_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_3_tensorarrayunstack_tensorlistfromtensor
rcnn_1_map_3_while_53432	
rcnn_1_map_3_while_53434	!
rcnn_1_map_3_while_rcnn_1_padÖ
Drcnn_1/map_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2F
Drcnn_1/map_3/while/TensorArrayV2Read/TensorListGetItem/element_shape
6rcnn_1/map_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmrcnn_1_map_3_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_3_tensorarrayunstack_tensorlistfromtensor_0rcnn_1_map_3_while_placeholderMrcnn_1/map_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:*
element_dtype0	28
6rcnn_1/map_3/while/TensorArrayV2Read/TensorListGetItemå
"rcnn_1/map_3/while/PartitionedCallPartitionedCall=rcnn_1/map_3/while/TensorArrayV2Read/TensorListGetItem:item:0rcnn_1_map_3_while_53432_0rcnn_1_map_3_while_53434_0rcnn_1_map_3_while_rcnn_1_pad_0*
Tin
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_getROIfeature8_534312$
"rcnn_1/map_3/while/PartitionedCall£
7rcnn_1/map_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem rcnn_1_map_3_while_placeholder_1rcnn_1_map_3_while_placeholder+rcnn_1/map_3/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype029
7rcnn_1/map_3/while/TensorArrayV2Write/TensorListSetItemv
rcnn_1/map_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map_3/while/add/y
rcnn_1/map_3/while/addAddV2rcnn_1_map_3_while_placeholder!rcnn_1/map_3/while/add/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map_3/while/addz
rcnn_1/map_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map_3/while/add_1/y·
rcnn_1/map_3/while/add_1AddV22rcnn_1_map_3_while_rcnn_1_map_3_while_loop_counter#rcnn_1/map_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map_3/while/add_1
rcnn_1/map_3/while/IdentityIdentityrcnn_1/map_3/while/add_1:z:0*
T0*
_output_shapes
: 2
rcnn_1/map_3/while/Identity
rcnn_1/map_3/while/Identity_1Identity-rcnn_1_map_3_while_rcnn_1_map_3_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_3/while/Identity_1
rcnn_1/map_3/while/Identity_2Identityrcnn_1/map_3/while/add:z:0*
T0*
_output_shapes
: 2
rcnn_1/map_3/while/Identity_2´
rcnn_1/map_3/while/Identity_3IdentityGrcnn_1/map_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
rcnn_1/map_3/while/Identity_3"6
rcnn_1_map_3_while_53432rcnn_1_map_3_while_53432_0"6
rcnn_1_map_3_while_53434rcnn_1_map_3_while_53434_0"C
rcnn_1_map_3_while_identity$rcnn_1/map_3/while/Identity:output:0"G
rcnn_1_map_3_while_identity_1&rcnn_1/map_3/while/Identity_1:output:0"G
rcnn_1_map_3_while_identity_2&rcnn_1/map_3/while/Identity_2:output:0"G
rcnn_1_map_3_while_identity_3&rcnn_1/map_3/while/Identity_3:output:0"d
/rcnn_1_map_3_while_rcnn_1_map_3_strided_slice_11rcnn_1_map_3_while_rcnn_1_map_3_strided_slice_1_0"@
rcnn_1_map_3_while_rcnn_1_padrcnn_1_map_3_while_rcnn_1_pad_0"Ü
krcnn_1_map_3_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_3_tensorarrayunstack_tensorlistfromtensormrcnn_1_map_3_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_3_tensorarrayunstack_tensorlistfromtensor_0*G
_input_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
Ê
¯
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_54631

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
«
K
/__inference_max_pooling2d_9_layer_call_fn_54465

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_544592
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_9_layer_call_and_return_conditional_losses_56842

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_55268

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64317

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¥
ª
7__inference_batch_normalization_198_layer_call_fn_65037

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_547782
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
ª
7__inference_batch_normalization_201_layer_call_fn_65481

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_551022
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ü
|
'__inference_dense_9_layer_call_fn_65636

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_568422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_197_layer_call_and_return_conditional_losses_63928

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:À*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_197_layer_call_fn_63937

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_197_layer_call_and_return_conditional_losses_552332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
ª
7__inference_batch_normalization_191_layer_call_fn_63988

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_552682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_54679

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ª
7__inference_batch_normalization_195_layer_call_fn_64644

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_544112
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
	
­
E__inference_conv2d_207_layer_call_and_return_conditional_losses_55509

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_200_layer_call_and_return_conditional_losses_64372

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs


*__inference_conv2d_203_layer_call_fn_64825

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_203_layer_call_and_return_conditional_losses_560562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_54411

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
±
«
C__inference_dense_10_layer_call_and_return_conditional_losses_56881

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65223

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_56365

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_55735

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
ª
7__inference_batch_normalization_199_layer_call_fn_65185

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_548822
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65371

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65519

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs


R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64631

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_54118

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_55544

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Û
ª
7__inference_batch_normalization_194_layer_call_fn_64496

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_562652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64169

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
h
 __inference_getROIfeature8_53431

inputs	
unknown	
	unknown_0	

rcnn_1_pad
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constß
PartitionedCallPartitionedCallinputsConst:output:0unknown	unknown_0
rcnn_1_pad*
Tin	
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference_getROIfeature_530382
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.::::ÿÿÿÿÿÿÿÿÿÀ:B >

_output_shapes
:
 
_user_specified_nameinputs:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ

¯
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_55168

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_196_layer_call_and_return_conditional_losses_63780

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_54558

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_63957

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÀ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64401

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_198_layer_call_and_return_conditional_losses_64076

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64549

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63809

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
ª
7__inference_batch_normalization_194_layer_call_fn_64509

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_562832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64335

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs


R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_54442

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ú
|
'__inference_dense_7_layer_call_fn_65596

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_567642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
ª
7__inference_batch_normalization_198_layer_call_fn_65101

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_557352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65289

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
ª
7__inference_batch_normalization_193_layer_call_fn_64361

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_566572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64483

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64253

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs


R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64863

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
óæ
5
!__inference__traced_restore_66233
file_prefix-
)assignvariableop_rcnn_1_conv2d_196_kernel-
)assignvariableop_1_rcnn_1_conv2d_196_bias;
7assignvariableop_2_rcnn_1_batch_normalization_190_gamma:
6assignvariableop_3_rcnn_1_batch_normalization_190_betaA
=assignvariableop_4_rcnn_1_batch_normalization_190_moving_meanE
Aassignvariableop_5_rcnn_1_batch_normalization_190_moving_variance/
+assignvariableop_6_rcnn_1_conv2d_197_kernel-
)assignvariableop_7_rcnn_1_conv2d_197_bias;
7assignvariableop_8_rcnn_1_batch_normalization_191_gamma:
6assignvariableop_9_rcnn_1_batch_normalization_191_betaB
>assignvariableop_10_rcnn_1_batch_normalization_191_moving_meanF
Bassignvariableop_11_rcnn_1_batch_normalization_191_moving_variance0
,assignvariableop_12_rcnn_1_conv2d_198_kernel.
*assignvariableop_13_rcnn_1_conv2d_198_bias<
8assignvariableop_14_rcnn_1_batch_normalization_192_gamma;
7assignvariableop_15_rcnn_1_batch_normalization_192_betaB
>assignvariableop_16_rcnn_1_batch_normalization_192_moving_meanF
Bassignvariableop_17_rcnn_1_batch_normalization_192_moving_variance0
,assignvariableop_18_rcnn_1_conv2d_199_kernel.
*assignvariableop_19_rcnn_1_conv2d_199_bias<
8assignvariableop_20_rcnn_1_batch_normalization_193_gamma;
7assignvariableop_21_rcnn_1_batch_normalization_193_betaB
>assignvariableop_22_rcnn_1_batch_normalization_193_moving_meanF
Bassignvariableop_23_rcnn_1_batch_normalization_193_moving_variance0
,assignvariableop_24_rcnn_1_conv2d_200_kernel.
*assignvariableop_25_rcnn_1_conv2d_200_bias<
8assignvariableop_26_rcnn_1_batch_normalization_194_gamma;
7assignvariableop_27_rcnn_1_batch_normalization_194_betaB
>assignvariableop_28_rcnn_1_batch_normalization_194_moving_meanF
Bassignvariableop_29_rcnn_1_batch_normalization_194_moving_variance0
,assignvariableop_30_rcnn_1_conv2d_201_kernel.
*assignvariableop_31_rcnn_1_conv2d_201_bias<
8assignvariableop_32_rcnn_1_batch_normalization_195_gamma;
7assignvariableop_33_rcnn_1_batch_normalization_195_betaB
>assignvariableop_34_rcnn_1_batch_normalization_195_moving_meanF
Bassignvariableop_35_rcnn_1_batch_normalization_195_moving_variance0
,assignvariableop_36_rcnn_1_conv2d_202_kernel.
*assignvariableop_37_rcnn_1_conv2d_202_bias<
8assignvariableop_38_rcnn_1_batch_normalization_196_gamma;
7assignvariableop_39_rcnn_1_batch_normalization_196_betaB
>assignvariableop_40_rcnn_1_batch_normalization_196_moving_meanF
Bassignvariableop_41_rcnn_1_batch_normalization_196_moving_variance0
,assignvariableop_42_rcnn_1_conv2d_203_kernel.
*assignvariableop_43_rcnn_1_conv2d_203_bias<
8assignvariableop_44_rcnn_1_batch_normalization_197_gamma;
7assignvariableop_45_rcnn_1_batch_normalization_197_betaB
>assignvariableop_46_rcnn_1_batch_normalization_197_moving_meanF
Bassignvariableop_47_rcnn_1_batch_normalization_197_moving_variance0
,assignvariableop_48_rcnn_1_conv2d_204_kernel.
*assignvariableop_49_rcnn_1_conv2d_204_bias<
8assignvariableop_50_rcnn_1_batch_normalization_198_gamma;
7assignvariableop_51_rcnn_1_batch_normalization_198_betaB
>assignvariableop_52_rcnn_1_batch_normalization_198_moving_meanF
Bassignvariableop_53_rcnn_1_batch_normalization_198_moving_variance0
,assignvariableop_54_rcnn_1_conv2d_205_kernel.
*assignvariableop_55_rcnn_1_conv2d_205_bias<
8assignvariableop_56_rcnn_1_batch_normalization_199_gamma;
7assignvariableop_57_rcnn_1_batch_normalization_199_betaB
>assignvariableop_58_rcnn_1_batch_normalization_199_moving_meanF
Bassignvariableop_59_rcnn_1_batch_normalization_199_moving_variance0
,assignvariableop_60_rcnn_1_conv2d_206_kernel.
*assignvariableop_61_rcnn_1_conv2d_206_bias<
8assignvariableop_62_rcnn_1_batch_normalization_200_gamma;
7assignvariableop_63_rcnn_1_batch_normalization_200_betaB
>assignvariableop_64_rcnn_1_batch_normalization_200_moving_meanF
Bassignvariableop_65_rcnn_1_batch_normalization_200_moving_variance0
,assignvariableop_66_rcnn_1_conv2d_207_kernel.
*assignvariableop_67_rcnn_1_conv2d_207_bias<
8assignvariableop_68_rcnn_1_batch_normalization_201_gamma;
7assignvariableop_69_rcnn_1_batch_normalization_201_betaB
>assignvariableop_70_rcnn_1_batch_normalization_201_moving_meanF
Bassignvariableop_71_rcnn_1_batch_normalization_201_moving_variance-
)assignvariableop_72_rcnn_1_dense_6_kernel+
'assignvariableop_73_rcnn_1_dense_6_bias-
)assignvariableop_74_rcnn_1_dense_7_kernel+
'assignvariableop_75_rcnn_1_dense_7_bias-
)assignvariableop_76_rcnn_1_dense_8_kernel+
'assignvariableop_77_rcnn_1_dense_8_bias-
)assignvariableop_78_rcnn_1_dense_9_kernel+
'assignvariableop_79_rcnn_1_dense_9_bias.
*assignvariableop_80_rcnn_1_dense_10_kernel,
(assignvariableop_81_rcnn_1_dense_10_bias.
*assignvariableop_82_rcnn_1_dense_11_kernel,
(assignvariableop_83_rcnn_1_dense_11_bias
identity_85¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_9%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*«$
value¡$B$UB2conv2d_condense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB0conv2d_condense1/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm1/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB2conv2d_condense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB0conv2d_condense2/bias/.ATTRIBUTES/VARIABLE_VALUEB,batch_norm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB+batch_norm2/beta/.ATTRIBUTES/VARIABLE_VALUEB2batch_norm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB;conv2d_extract12_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB9conv2d_extract12_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm_extract12a/gamma/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract12a/beta/.ATTRIBUTES/VARIABLE_VALUEB<batch_norm_extract12a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@batch_norm_extract12a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB2conv2d_extract12/kernel/.ATTRIBUTES/VARIABLE_VALUEB0conv2d_extract12/bias/.ATTRIBUTES/VARIABLE_VALUEB6batch_norm_extract12b/gamma/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract12b/beta/.ATTRIBUTES/VARIABLE_VALUEB<batch_norm_extract12b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@batch_norm_extract12b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB:conv2d_extract8_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB8conv2d_extract8_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract8a/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract8a/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract8a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract8a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1conv2d_extract8/kernel/.ATTRIBUTES/VARIABLE_VALUEB/conv2d_extract8/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract8b/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract8b/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract8b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract8b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB:conv2d_extract5_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB8conv2d_extract5_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract5a/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract5a/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract5a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract5a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1conv2d_extract5/kernel/.ATTRIBUTES/VARIABLE_VALUEB/conv2d_extract5/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract5b/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract5b/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract5b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract5b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB:conv2d_extract3_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB8conv2d_extract3_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract3a/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract3a/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract3a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract3a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1conv2d_extract3/kernel/.ATTRIBUTES/VARIABLE_VALUEB/conv2d_extract3/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract3b/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract3b/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract3b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract3b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB:conv2d_extract2_condense/kernel/.ATTRIBUTES/VARIABLE_VALUEB8conv2d_extract2_condense/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract2a/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract2a/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract2a/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract2a/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1conv2d_extract2/kernel/.ATTRIBUTES/VARIABLE_VALUEB/conv2d_extract2/bias/.ATTRIBUTES/VARIABLE_VALUEB5batch_norm_extract2b/gamma/.ATTRIBUTES/VARIABLE_VALUEB4batch_norm_extract2b/beta/.ATTRIBUTES/VARIABLE_VALUEB;batch_norm_extract2b/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?batch_norm_extract2b/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB-classifier1/kernel/.ATTRIBUTES/VARIABLE_VALUEB+classifier1/bias/.ATTRIBUTES/VARIABLE_VALUEB-classifier2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+classifier2/bias/.ATTRIBUTES/VARIABLE_VALUEB-classifier3/kernel/.ATTRIBUTES/VARIABLE_VALUEB+classifier3/bias/.ATTRIBUTES/VARIABLE_VALUEB,regressor1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*regressor1/bias/.ATTRIBUTES/VARIABLE_VALUEB,regressor2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*regressor2/bias/.ATTRIBUTES/VARIABLE_VALUEB,regressor3/kernel/.ATTRIBUTES/VARIABLE_VALUEB*regressor3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names»
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:U*
dtype0*¿
valueµB²UB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices×
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ê
_output_shapes×
Ô:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*c
dtypesY
W2U2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¨
AssignVariableOpAssignVariableOp)assignvariableop_rcnn_1_conv2d_196_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1®
AssignVariableOp_1AssignVariableOp)assignvariableop_1_rcnn_1_conv2d_196_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¼
AssignVariableOp_2AssignVariableOp7assignvariableop_2_rcnn_1_batch_normalization_190_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3»
AssignVariableOp_3AssignVariableOp6assignvariableop_3_rcnn_1_batch_normalization_190_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Â
AssignVariableOp_4AssignVariableOp=assignvariableop_4_rcnn_1_batch_normalization_190_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Æ
AssignVariableOp_5AssignVariableOpAassignvariableop_5_rcnn_1_batch_normalization_190_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6°
AssignVariableOp_6AssignVariableOp+assignvariableop_6_rcnn_1_conv2d_197_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7®
AssignVariableOp_7AssignVariableOp)assignvariableop_7_rcnn_1_conv2d_197_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¼
AssignVariableOp_8AssignVariableOp7assignvariableop_8_rcnn_1_batch_normalization_191_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9»
AssignVariableOp_9AssignVariableOp6assignvariableop_9_rcnn_1_batch_normalization_191_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Æ
AssignVariableOp_10AssignVariableOp>assignvariableop_10_rcnn_1_batch_normalization_191_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ê
AssignVariableOp_11AssignVariableOpBassignvariableop_11_rcnn_1_batch_normalization_191_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12´
AssignVariableOp_12AssignVariableOp,assignvariableop_12_rcnn_1_conv2d_198_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13²
AssignVariableOp_13AssignVariableOp*assignvariableop_13_rcnn_1_conv2d_198_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14À
AssignVariableOp_14AssignVariableOp8assignvariableop_14_rcnn_1_batch_normalization_192_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¿
AssignVariableOp_15AssignVariableOp7assignvariableop_15_rcnn_1_batch_normalization_192_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Æ
AssignVariableOp_16AssignVariableOp>assignvariableop_16_rcnn_1_batch_normalization_192_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ê
AssignVariableOp_17AssignVariableOpBassignvariableop_17_rcnn_1_batch_normalization_192_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18´
AssignVariableOp_18AssignVariableOp,assignvariableop_18_rcnn_1_conv2d_199_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19²
AssignVariableOp_19AssignVariableOp*assignvariableop_19_rcnn_1_conv2d_199_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20À
AssignVariableOp_20AssignVariableOp8assignvariableop_20_rcnn_1_batch_normalization_193_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¿
AssignVariableOp_21AssignVariableOp7assignvariableop_21_rcnn_1_batch_normalization_193_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Æ
AssignVariableOp_22AssignVariableOp>assignvariableop_22_rcnn_1_batch_normalization_193_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ê
AssignVariableOp_23AssignVariableOpBassignvariableop_23_rcnn_1_batch_normalization_193_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24´
AssignVariableOp_24AssignVariableOp,assignvariableop_24_rcnn_1_conv2d_200_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_rcnn_1_conv2d_200_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26À
AssignVariableOp_26AssignVariableOp8assignvariableop_26_rcnn_1_batch_normalization_194_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¿
AssignVariableOp_27AssignVariableOp7assignvariableop_27_rcnn_1_batch_normalization_194_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Æ
AssignVariableOp_28AssignVariableOp>assignvariableop_28_rcnn_1_batch_normalization_194_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ê
AssignVariableOp_29AssignVariableOpBassignvariableop_29_rcnn_1_batch_normalization_194_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30´
AssignVariableOp_30AssignVariableOp,assignvariableop_30_rcnn_1_conv2d_201_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31²
AssignVariableOp_31AssignVariableOp*assignvariableop_31_rcnn_1_conv2d_201_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32À
AssignVariableOp_32AssignVariableOp8assignvariableop_32_rcnn_1_batch_normalization_195_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¿
AssignVariableOp_33AssignVariableOp7assignvariableop_33_rcnn_1_batch_normalization_195_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Æ
AssignVariableOp_34AssignVariableOp>assignvariableop_34_rcnn_1_batch_normalization_195_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ê
AssignVariableOp_35AssignVariableOpBassignvariableop_35_rcnn_1_batch_normalization_195_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36´
AssignVariableOp_36AssignVariableOp,assignvariableop_36_rcnn_1_conv2d_202_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_rcnn_1_conv2d_202_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38À
AssignVariableOp_38AssignVariableOp8assignvariableop_38_rcnn_1_batch_normalization_196_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¿
AssignVariableOp_39AssignVariableOp7assignvariableop_39_rcnn_1_batch_normalization_196_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Æ
AssignVariableOp_40AssignVariableOp>assignvariableop_40_rcnn_1_batch_normalization_196_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ê
AssignVariableOp_41AssignVariableOpBassignvariableop_41_rcnn_1_batch_normalization_196_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42´
AssignVariableOp_42AssignVariableOp,assignvariableop_42_rcnn_1_conv2d_203_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_rcnn_1_conv2d_203_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44À
AssignVariableOp_44AssignVariableOp8assignvariableop_44_rcnn_1_batch_normalization_197_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¿
AssignVariableOp_45AssignVariableOp7assignvariableop_45_rcnn_1_batch_normalization_197_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Æ
AssignVariableOp_46AssignVariableOp>assignvariableop_46_rcnn_1_batch_normalization_197_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ê
AssignVariableOp_47AssignVariableOpBassignvariableop_47_rcnn_1_batch_normalization_197_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48´
AssignVariableOp_48AssignVariableOp,assignvariableop_48_rcnn_1_conv2d_204_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49²
AssignVariableOp_49AssignVariableOp*assignvariableop_49_rcnn_1_conv2d_204_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50À
AssignVariableOp_50AssignVariableOp8assignvariableop_50_rcnn_1_batch_normalization_198_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¿
AssignVariableOp_51AssignVariableOp7assignvariableop_51_rcnn_1_batch_normalization_198_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Æ
AssignVariableOp_52AssignVariableOp>assignvariableop_52_rcnn_1_batch_normalization_198_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ê
AssignVariableOp_53AssignVariableOpBassignvariableop_53_rcnn_1_batch_normalization_198_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54´
AssignVariableOp_54AssignVariableOp,assignvariableop_54_rcnn_1_conv2d_205_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55²
AssignVariableOp_55AssignVariableOp*assignvariableop_55_rcnn_1_conv2d_205_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56À
AssignVariableOp_56AssignVariableOp8assignvariableop_56_rcnn_1_batch_normalization_199_gammaIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¿
AssignVariableOp_57AssignVariableOp7assignvariableop_57_rcnn_1_batch_normalization_199_betaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Æ
AssignVariableOp_58AssignVariableOp>assignvariableop_58_rcnn_1_batch_normalization_199_moving_meanIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ê
AssignVariableOp_59AssignVariableOpBassignvariableop_59_rcnn_1_batch_normalization_199_moving_varianceIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60´
AssignVariableOp_60AssignVariableOp,assignvariableop_60_rcnn_1_conv2d_206_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61²
AssignVariableOp_61AssignVariableOp*assignvariableop_61_rcnn_1_conv2d_206_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62À
AssignVariableOp_62AssignVariableOp8assignvariableop_62_rcnn_1_batch_normalization_200_gammaIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¿
AssignVariableOp_63AssignVariableOp7assignvariableop_63_rcnn_1_batch_normalization_200_betaIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Æ
AssignVariableOp_64AssignVariableOp>assignvariableop_64_rcnn_1_batch_normalization_200_moving_meanIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ê
AssignVariableOp_65AssignVariableOpBassignvariableop_65_rcnn_1_batch_normalization_200_moving_varianceIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66´
AssignVariableOp_66AssignVariableOp,assignvariableop_66_rcnn_1_conv2d_207_kernelIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67²
AssignVariableOp_67AssignVariableOp*assignvariableop_67_rcnn_1_conv2d_207_biasIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68À
AssignVariableOp_68AssignVariableOp8assignvariableop_68_rcnn_1_batch_normalization_201_gammaIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¿
AssignVariableOp_69AssignVariableOp7assignvariableop_69_rcnn_1_batch_normalization_201_betaIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Æ
AssignVariableOp_70AssignVariableOp>assignvariableop_70_rcnn_1_batch_normalization_201_moving_meanIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ê
AssignVariableOp_71AssignVariableOpBassignvariableop_71_rcnn_1_batch_normalization_201_moving_varianceIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72±
AssignVariableOp_72AssignVariableOp)assignvariableop_72_rcnn_1_dense_6_kernelIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73¯
AssignVariableOp_73AssignVariableOp'assignvariableop_73_rcnn_1_dense_6_biasIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74±
AssignVariableOp_74AssignVariableOp)assignvariableop_74_rcnn_1_dense_7_kernelIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75¯
AssignVariableOp_75AssignVariableOp'assignvariableop_75_rcnn_1_dense_7_biasIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76±
AssignVariableOp_76AssignVariableOp)assignvariableop_76_rcnn_1_dense_8_kernelIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77¯
AssignVariableOp_77AssignVariableOp'assignvariableop_77_rcnn_1_dense_8_biasIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78±
AssignVariableOp_78AssignVariableOp)assignvariableop_78_rcnn_1_dense_9_kernelIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79¯
AssignVariableOp_79AssignVariableOp'assignvariableop_79_rcnn_1_dense_9_biasIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80²
AssignVariableOp_80AssignVariableOp*assignvariableop_80_rcnn_1_dense_10_kernelIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81°
AssignVariableOp_81AssignVariableOp(assignvariableop_81_rcnn_1_dense_10_biasIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82²
AssignVariableOp_82AssignVariableOp*assignvariableop_82_rcnn_1_dense_11_kernelIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83°
AssignVariableOp_83AssignVariableOp(assignvariableop_83_rcnn_1_dense_11_biasIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_839
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_84Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_84
Identity_85IdentityIdentity_84:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_85"#
identity_85Identity_85:output:0*ç
_input_shapesÕ
Ò: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ê
¯
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_55071

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý
ª
7__inference_batch_normalization_200_layer_call_fn_65397

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_554622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_56283

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64715

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
ª
7__inference_batch_normalization_192_layer_call_fn_64136

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_565392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
ª
7__inference_batch_normalization_194_layer_call_fn_64445

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_543382
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó#
ò
rcnn_1_map_1_while_body_531436
2rcnn_1_map_1_while_rcnn_1_map_1_while_loop_counter1
-rcnn_1_map_1_while_rcnn_1_map_1_strided_slice"
rcnn_1_map_1_while_placeholder$
 rcnn_1_map_1_while_placeholder_15
1rcnn_1_map_1_while_rcnn_1_map_1_strided_slice_1_0q
mrcnn_1_map_1_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_1_tensorarrayunstack_tensorlistfromtensor_0
rcnn_1_map_1_while_53176_0	
rcnn_1_map_1_while_53178_0	#
rcnn_1_map_1_while_rcnn_1_pad_0
rcnn_1_map_1_while_identity!
rcnn_1_map_1_while_identity_1!
rcnn_1_map_1_while_identity_2!
rcnn_1_map_1_while_identity_33
/rcnn_1_map_1_while_rcnn_1_map_1_strided_slice_1o
krcnn_1_map_1_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_1_tensorarrayunstack_tensorlistfromtensor
rcnn_1_map_1_while_53176	
rcnn_1_map_1_while_53178	!
rcnn_1_map_1_while_rcnn_1_padÖ
Drcnn_1/map_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2F
Drcnn_1/map_1/while/TensorArrayV2Read/TensorListGetItem/element_shape
6rcnn_1/map_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmrcnn_1_map_1_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_1_tensorarrayunstack_tensorlistfromtensor_0rcnn_1_map_1_while_placeholderMrcnn_1/map_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:*
element_dtype0	28
6rcnn_1/map_1/while/TensorArrayV2Read/TensorListGetItemå
"rcnn_1/map_1/while/PartitionedCallPartitionedCall=rcnn_1/map_1/while/TensorArrayV2Read/TensorListGetItem:item:0rcnn_1_map_1_while_53176_0rcnn_1_map_1_while_53178_0rcnn_1_map_1_while_rcnn_1_pad_0*
Tin
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_getROIfeature3_531752$
"rcnn_1/map_1/while/PartitionedCall£
7rcnn_1/map_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem rcnn_1_map_1_while_placeholder_1rcnn_1_map_1_while_placeholder+rcnn_1/map_1/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype029
7rcnn_1/map_1/while/TensorArrayV2Write/TensorListSetItemv
rcnn_1/map_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map_1/while/add/y
rcnn_1/map_1/while/addAddV2rcnn_1_map_1_while_placeholder!rcnn_1/map_1/while/add/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map_1/while/addz
rcnn_1/map_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map_1/while/add_1/y·
rcnn_1/map_1/while/add_1AddV22rcnn_1_map_1_while_rcnn_1_map_1_while_loop_counter#rcnn_1/map_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map_1/while/add_1
rcnn_1/map_1/while/IdentityIdentityrcnn_1/map_1/while/add_1:z:0*
T0*
_output_shapes
: 2
rcnn_1/map_1/while/Identity
rcnn_1/map_1/while/Identity_1Identity-rcnn_1_map_1_while_rcnn_1_map_1_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_1/while/Identity_1
rcnn_1/map_1/while/Identity_2Identityrcnn_1/map_1/while/add:z:0*
T0*
_output_shapes
: 2
rcnn_1/map_1/while/Identity_2´
rcnn_1/map_1/while/Identity_3IdentityGrcnn_1/map_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
rcnn_1/map_1/while/Identity_3"6
rcnn_1_map_1_while_53176rcnn_1_map_1_while_53176_0"6
rcnn_1_map_1_while_53178rcnn_1_map_1_while_53178_0"C
rcnn_1_map_1_while_identity$rcnn_1/map_1/while/Identity:output:0"G
rcnn_1_map_1_while_identity_1&rcnn_1/map_1/while/Identity_1:output:0"G
rcnn_1_map_1_while_identity_2&rcnn_1/map_1/while/Identity_2:output:0"G
rcnn_1_map_1_while_identity_3&rcnn_1/map_1/while/Identity_3:output:0"d
/rcnn_1_map_1_while_rcnn_1_map_1_strided_slice_11rcnn_1_map_1_while_rcnn_1_map_1_strided_slice_1_0"@
rcnn_1_map_1_while_rcnn_1_padrcnn_1_map_1_while_rcnn_1_pad_0"Ü
krcnn_1_map_1_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_1_tensorarrayunstack_tensorlistfromtensormrcnn_1_map_1_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_1_tensorarrayunstack_tensorlistfromtensor_0*G
_input_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
¡
ª
7__inference_batch_normalization_195_layer_call_fn_64657

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_544422
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
«
K
/__inference_max_pooling2d_8_layer_call_fn_54245

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_542392
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_54191

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64187

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_56639

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_56557

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_196_layer_call_fn_63789

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_196_layer_call_and_return_conditional_losses_551332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_56265

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô#
ò
rcnn_1_map_4_while_body_535276
2rcnn_1_map_4_while_rcnn_1_map_4_while_loop_counter1
-rcnn_1_map_4_while_rcnn_1_map_4_strided_slice"
rcnn_1_map_4_while_placeholder$
 rcnn_1_map_4_while_placeholder_15
1rcnn_1_map_4_while_rcnn_1_map_4_strided_slice_1_0q
mrcnn_1_map_4_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_4_tensorarrayunstack_tensorlistfromtensor_0
rcnn_1_map_4_while_53560_0	
rcnn_1_map_4_while_53562_0	#
rcnn_1_map_4_while_rcnn_1_pad_0
rcnn_1_map_4_while_identity!
rcnn_1_map_4_while_identity_1!
rcnn_1_map_4_while_identity_2!
rcnn_1_map_4_while_identity_33
/rcnn_1_map_4_while_rcnn_1_map_4_strided_slice_1o
krcnn_1_map_4_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_4_tensorarrayunstack_tensorlistfromtensor
rcnn_1_map_4_while_53560	
rcnn_1_map_4_while_53562	!
rcnn_1_map_4_while_rcnn_1_padÖ
Drcnn_1/map_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2F
Drcnn_1/map_4/while/TensorArrayV2Read/TensorListGetItem/element_shape
6rcnn_1/map_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmrcnn_1_map_4_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_4_tensorarrayunstack_tensorlistfromtensor_0rcnn_1_map_4_while_placeholderMrcnn_1/map_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:*
element_dtype0	28
6rcnn_1/map_4/while/TensorArrayV2Read/TensorListGetItemæ
"rcnn_1/map_4/while/PartitionedCallPartitionedCall=rcnn_1/map_4/while/TensorArrayV2Read/TensorListGetItem:item:0rcnn_1_map_4_while_53560_0rcnn_1_map_4_while_53562_0rcnn_1_map_4_while_rcnn_1_pad_0*
Tin
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_getROIfeature12_535592$
"rcnn_1/map_4/while/PartitionedCall£
7rcnn_1/map_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem rcnn_1_map_4_while_placeholder_1rcnn_1_map_4_while_placeholder+rcnn_1/map_4/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype029
7rcnn_1/map_4/while/TensorArrayV2Write/TensorListSetItemv
rcnn_1/map_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map_4/while/add/y
rcnn_1/map_4/while/addAddV2rcnn_1_map_4_while_placeholder!rcnn_1/map_4/while/add/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map_4/while/addz
rcnn_1/map_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map_4/while/add_1/y·
rcnn_1/map_4/while/add_1AddV22rcnn_1_map_4_while_rcnn_1_map_4_while_loop_counter#rcnn_1/map_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map_4/while/add_1
rcnn_1/map_4/while/IdentityIdentityrcnn_1/map_4/while/add_1:z:0*
T0*
_output_shapes
: 2
rcnn_1/map_4/while/Identity
rcnn_1/map_4/while/Identity_1Identity-rcnn_1_map_4_while_rcnn_1_map_4_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_4/while/Identity_1
rcnn_1/map_4/while/Identity_2Identityrcnn_1/map_4/while/add:z:0*
T0*
_output_shapes
: 2
rcnn_1/map_4/while/Identity_2´
rcnn_1/map_4/while/Identity_3IdentityGrcnn_1/map_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
rcnn_1/map_4/while/Identity_3"6
rcnn_1_map_4_while_53560rcnn_1_map_4_while_53560_0"6
rcnn_1_map_4_while_53562rcnn_1_map_4_while_53562_0"C
rcnn_1_map_4_while_identity$rcnn_1/map_4/while/Identity:output:0"G
rcnn_1_map_4_while_identity_1&rcnn_1/map_4/while/Identity_1:output:0"G
rcnn_1_map_4_while_identity_2&rcnn_1/map_4/while/Identity_2:output:0"G
rcnn_1_map_4_while_identity_3&rcnn_1/map_4/while/Identity_3:output:0"d
/rcnn_1_map_4_while_rcnn_1_map_4_strided_slice_11rcnn_1_map_4_while_rcnn_1_map_4_strided_slice_1_0"@
rcnn_1_map_4_while_rcnn_1_padrcnn_1_map_4_while_rcnn_1_pad_0"Ü
krcnn_1_map_4_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_4_tensorarrayunstack_tensorlistfromtensormrcnn_1_map_4_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_4_tensorarrayunstack_tensorlistfromtensor_0*G
_input_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
¦

R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_54014

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:À*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:À*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:À*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:À:À:À:À:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_54527

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_204_layer_call_and_return_conditional_losses_55682

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:À*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÀ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs

ª
7__inference_batch_normalization_199_layer_call_fn_65172

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_548512
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_54967

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
h
 __inference_getROIfeature2_53046

inputs	
unknown	
	unknown_0	

rcnn_1_pad
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constß
PartitionedCallPartitionedCallinputsConst:output:0unknown	unknown_0
rcnn_1_pad*
Tin	
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference_getROIfeature_530382
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.::::ÿÿÿÿÿÿÿÿÿÀ:B >

_output_shapes
:
 
_user_specified_nameinputs:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
±
«
C__inference_dense_10_layer_call_and_return_conditional_losses_65647

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_54851

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ù
i
!__inference_getROIfeature12_53559

inputs	
unknown	
	unknown_0	

rcnn_1_pad
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constß
PartitionedCallPartitionedCallinputsConst:output:0unknown	unknown_0
rcnn_1_pad*
Tin	
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference_getROIfeature_530382
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.::::ÿÿÿÿÿÿÿÿÿÀ:B >

_output_shapes
:
 
_user_specified_nameinputs:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
×
ª
7__inference_batch_normalization_195_layer_call_fn_64580

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_563652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ó#
ò
rcnn_1_map_2_while_body_532716
2rcnn_1_map_2_while_rcnn_1_map_2_while_loop_counter1
-rcnn_1_map_2_while_rcnn_1_map_2_strided_slice"
rcnn_1_map_2_while_placeholder$
 rcnn_1_map_2_while_placeholder_15
1rcnn_1_map_2_while_rcnn_1_map_2_strided_slice_1_0q
mrcnn_1_map_2_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_2_tensorarrayunstack_tensorlistfromtensor_0
rcnn_1_map_2_while_53304_0	
rcnn_1_map_2_while_53306_0	#
rcnn_1_map_2_while_rcnn_1_pad_0
rcnn_1_map_2_while_identity!
rcnn_1_map_2_while_identity_1!
rcnn_1_map_2_while_identity_2!
rcnn_1_map_2_while_identity_33
/rcnn_1_map_2_while_rcnn_1_map_2_strided_slice_1o
krcnn_1_map_2_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_2_tensorarrayunstack_tensorlistfromtensor
rcnn_1_map_2_while_53304	
rcnn_1_map_2_while_53306	!
rcnn_1_map_2_while_rcnn_1_padÖ
Drcnn_1/map_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:2F
Drcnn_1/map_2/while/TensorArrayV2Read/TensorListGetItem/element_shape
6rcnn_1/map_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmrcnn_1_map_2_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_2_tensorarrayunstack_tensorlistfromtensor_0rcnn_1_map_2_while_placeholderMrcnn_1/map_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:*
element_dtype0	28
6rcnn_1/map_2/while/TensorArrayV2Read/TensorListGetItemå
"rcnn_1/map_2/while/PartitionedCallPartitionedCall=rcnn_1/map_2/while/TensorArrayV2Read/TensorListGetItem:item:0rcnn_1_map_2_while_53304_0rcnn_1_map_2_while_53306_0rcnn_1_map_2_while_rcnn_1_pad_0*
Tin
2			*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_getROIfeature5_533032$
"rcnn_1/map_2/while/PartitionedCall£
7rcnn_1/map_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem rcnn_1_map_2_while_placeholder_1rcnn_1_map_2_while_placeholder+rcnn_1/map_2/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype029
7rcnn_1/map_2/while/TensorArrayV2Write/TensorListSetItemv
rcnn_1/map_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map_2/while/add/y
rcnn_1/map_2/while/addAddV2rcnn_1_map_2_while_placeholder!rcnn_1/map_2/while/add/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map_2/while/addz
rcnn_1/map_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rcnn_1/map_2/while/add_1/y·
rcnn_1/map_2/while/add_1AddV22rcnn_1_map_2_while_rcnn_1_map_2_while_loop_counter#rcnn_1/map_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rcnn_1/map_2/while/add_1
rcnn_1/map_2/while/IdentityIdentityrcnn_1/map_2/while/add_1:z:0*
T0*
_output_shapes
: 2
rcnn_1/map_2/while/Identity
rcnn_1/map_2/while/Identity_1Identity-rcnn_1_map_2_while_rcnn_1_map_2_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_2/while/Identity_1
rcnn_1/map_2/while/Identity_2Identityrcnn_1/map_2/while/add:z:0*
T0*
_output_shapes
: 2
rcnn_1/map_2/while/Identity_2´
rcnn_1/map_2/while/Identity_3IdentityGrcnn_1/map_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
rcnn_1/map_2/while/Identity_3"6
rcnn_1_map_2_while_53304rcnn_1_map_2_while_53304_0"6
rcnn_1_map_2_while_53306rcnn_1_map_2_while_53306_0"C
rcnn_1_map_2_while_identity$rcnn_1/map_2/while/Identity:output:0"G
rcnn_1_map_2_while_identity_1&rcnn_1/map_2/while/Identity_1:output:0"G
rcnn_1_map_2_while_identity_2&rcnn_1/map_2/while/Identity_2:output:0"G
rcnn_1_map_2_while_identity_3&rcnn_1/map_2/while/Identity_3:output:0"d
/rcnn_1_map_2_while_rcnn_1_map_2_strided_slice_11rcnn_1_map_2_while_rcnn_1_map_2_strided_slice_1_0"@
rcnn_1_map_2_while_rcnn_1_padrcnn_1_map_2_while_rcnn_1_pad_0"Ü
krcnn_1_map_2_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_2_tensorarrayunstack_tensorlistfromtensormrcnn_1_map_2_while_tensorarrayv2read_tensorlistgetitem_rcnn_1_map_2_tensorarrayunstack_tensorlistfromtensor_0*G
_input_shapes6
4: : : : : : :::ÿÿÿÿÿÿÿÿÿÀ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
	
­
E__inference_conv2d_201_layer_call_and_return_conditional_losses_56330

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
­
E__inference_conv2d_203_layer_call_and_return_conditional_losses_56056

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65159

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63891

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

é
rcnn_1_map_2_while_cond_532706
2rcnn_1_map_2_while_rcnn_1_map_2_while_loop_counter1
-rcnn_1_map_2_while_rcnn_1_map_2_strided_slice"
rcnn_1_map_2_while_placeholder$
 rcnn_1_map_2_while_placeholder_16
2rcnn_1_map_2_while_less_rcnn_1_map_2_strided_sliceM
Ircnn_1_map_2_while_rcnn_1_map_2_while_cond_53270___redundant_placeholder0M
Ircnn_1_map_2_while_rcnn_1_map_2_while_cond_53270___redundant_placeholder1	M
Ircnn_1_map_2_while_rcnn_1_map_2_while_cond_53270___redundant_placeholder2	M
Ircnn_1_map_2_while_rcnn_1_map_2_while_cond_53270___redundant_placeholder3
rcnn_1_map_2_while_identity
¯
rcnn_1/map_2/while/LessLessrcnn_1_map_2_while_placeholder2rcnn_1_map_2_while_less_rcnn_1_map_2_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_2/while/LessÂ
rcnn_1/map_2/while/Less_1Less2rcnn_1_map_2_while_rcnn_1_map_2_while_loop_counter-rcnn_1_map_2_while_rcnn_1_map_2_strided_slice*
T0*
_output_shapes
: 2
rcnn_1/map_2/while/Less_1 
rcnn_1/map_2/while/LogicalAnd
LogicalAndrcnn_1/map_2/while/Less_1:z:0rcnn_1/map_2/while/Less:z:0*
_output_shapes
: 2
rcnn_1/map_2/while/LogicalAnd
rcnn_1/map_2/while/IdentityIdentity!rcnn_1/map_2/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
rcnn_1/map_2/while/Identity"C
rcnn_1_map_2_while_identity$rcnn_1/map_2/while/Identity:output:0*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
£
ª
7__inference_batch_normalization_200_layer_call_fn_65320

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_549672
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ª
7__inference_batch_normalization_201_layer_call_fn_65468

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_550712
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_55717

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_54662

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¨
serving_default
D
input_19
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
G
	input_2_1:
serving_default_input_2_1:0ÿÿÿÿÿÿÿÿÿ
G
	input_2_2:
serving_default_input_2_2:0ÿÿÿÿÿÿÿÿÿ
G
	input_2_3:
serving_default_input_2_3:0ÿÿÿÿÿÿÿÿÿ
G
	input_2_4:
serving_default_input_2_4:0ÿÿÿÿÿÿÿÿÿ
G
	input_2_5:
serving_default_input_2_5:0ÿÿÿÿÿÿÿÿÿ>

output_1_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ>

output_1_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ>

output_1_30
StatefulPartitionedCall:2	ÿÿÿÿÿÿÿÿÿ>

output_2_10
StatefulPartitionedCall:3ÿÿÿÿÿÿÿÿÿ>

output_2_20
StatefulPartitionedCall:4ÿÿÿÿÿÿÿÿÿ>

output_2_30
StatefulPartitionedCall:5	ÿÿÿÿÿÿÿÿÿ>

output_3_10
StatefulPartitionedCall:6ÿÿÿÿÿÿÿÿÿ>

output_3_20
StatefulPartitionedCall:7ÿÿÿÿÿÿÿÿÿ>

output_3_30
StatefulPartitionedCall:8	ÿÿÿÿÿÿÿÿÿ>

output_4_10
StatefulPartitionedCall:9ÿÿÿÿÿÿÿÿÿ?

output_4_21
StatefulPartitionedCall:10ÿÿÿÿÿÿÿÿÿ?

output_4_31
StatefulPartitionedCall:11	ÿÿÿÿÿÿÿÿÿ?

output_5_11
StatefulPartitionedCall:12ÿÿÿÿÿÿÿÿÿ?

output_5_21
StatefulPartitionedCall:13ÿÿÿÿÿÿÿÿÿ?

output_5_31
StatefulPartitionedCall:14	ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÇÓ
Ó
conv2d_condense1
batch_norm1
conv2d_condense2
batch_norm2
conv2d_extract12_condense
batch_norm_extract12a
conv2d_extract12
batch_norm_extract12b
	maxpool_extract17_3

conv2d_extract8_condense
batch_norm_extract8a
conv2d_extract8
batch_norm_extract8b
maxpool_extract12_3
conv2d_extract5_condense
batch_norm_extract5a
conv2d_extract5
batch_norm_extract5b
maxpool_extract8_3
conv2d_extract3_condense
batch_norm_extract3a
conv2d_extract3
batch_norm_extract3b
maxpool_extract5_3
conv2d_extract2_condense
batch_norm_extract2a
conv2d_extract2
batch_norm_extract2b
flatten
classifier1
classifier2
 classifier3
!
regressor1
"
regressor2
#
regressor3
$	variables
%regularization_losses
&trainable_variables
'	keras_api
(
signatures
+É&call_and_return_all_conditional_losses
Ê__call__
Ë_default_save_signature
ÌgetROIfeature
ÍgetROIfeature12
ÎgetROIfeature2
ÏgetROIfeature3
ÐgetROIfeature5
ÑgetROIfeature8"ï
_tf_keras_modelÕ{"class_name": "RCNN", "name": "rcnn_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "RCNN"}}
ø	

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+Ò&call_and_return_all_conditional_losses
Ó__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_196", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_196", "trainable": true, "dtype": "float32", "filters": 384, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 768}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 17, 17, 768]}}
¿	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+Ô&call_and_return_all_conditional_losses
Õ__call__"é
_tf_keras_layerÏ{"class_name": "BatchNormalization", "name": "batch_normalization_190", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_190", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 384}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 17, 17, 384]}}
ø	

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+Ö&call_and_return_all_conditional_losses
×__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_197", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_197", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 384}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 17, 17, 384]}}
¿	
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+Ø&call_and_return_all_conditional_losses
Ù__call__"é
_tf_keras_layerÏ{"class_name": "BatchNormalization", "name": "batch_normalization_191", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_191", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 17, 17, 192]}}
ù	

Gkernel
Hbias
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
+Ú&call_and_return_all_conditional_losses
Û__call__"Ò
_tf_keras_layer¸{"class_name": "Conv2D", "name": "conv2d_198", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_198", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 17, 17, 192]}}
¿	
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+Ü&call_and_return_all_conditional_losses
Ý__call__"é
_tf_keras_layerÏ{"class_name": "BatchNormalization", "name": "batch_normalization_192", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_192", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 17, 17, 144]}}
ø	

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
+Þ&call_and_return_all_conditional_losses
ß__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_199", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_199", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 17, 17, 144]}}
»	
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+à&call_and_return_all_conditional_losses
á__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_193", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_193", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 3, 3, 48]}}

e	variables
fregularization_losses
gtrainable_variables
h	keras_api
+â&call_and_return_all_conditional_losses
ã__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [9, 9]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ù	

ikernel
jbias
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
+ä&call_and_return_all_conditional_losses
å__call__"Ò
_tf_keras_layer¸{"class_name": "Conv2D", "name": "conv2d_200", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_200", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 12, 12, 192]}}
¿	
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
+æ&call_and_return_all_conditional_losses
ç__call__"é
_tf_keras_layerÏ{"class_name": "BatchNormalization", "name": "batch_normalization_194", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_194", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 12, 12, 144]}}
ø	

xkernel
ybias
z	variables
{regularization_losses
|trainable_variables
}	keras_api
+è&call_and_return_all_conditional_losses
é__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_201", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_201", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 12, 12, 144]}}
Â	
~axis
	gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
+ê&call_and_return_all_conditional_losses
ë__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_195", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_195", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 3, 3, 48]}}

	variables
regularization_losses
trainable_variables
	keras_api
+ì&call_and_return_all_conditional_losses
í__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [6, 6]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ý	
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+î&call_and_return_all_conditional_losses
ï__call__"Ð
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_202", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_202", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 8, 8, 192]}}
Æ	
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
+ð&call_and_return_all_conditional_losses
ñ__call__"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_196", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_196", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 8, 8, 144]}}
ü	
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+ò&call_and_return_all_conditional_losses
ó__call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv2d_203", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_203", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 8, 8, 144]}}
Ä	
	 axis

¡gamma
	¢beta
£moving_mean
¤moving_variance
¥	variables
¦regularization_losses
§trainable_variables
¨	keras_api
+ô&call_and_return_all_conditional_losses
õ__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_197", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_197", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 3, 3, 48]}}

©	variables
ªregularization_losses
«trainable_variables
¬	keras_api
+ö&call_and_return_all_conditional_losses
÷__call__"ò
_tf_keras_layerØ{"class_name": "MaxPooling2D", "name": "max_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ý	
­kernel
	®bias
¯	variables
°regularization_losses
±trainable_variables
²	keras_api
+ø&call_and_return_all_conditional_losses
ù__call__"Ð
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_204", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_204", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 5, 5, 192]}}
Æ	
	³axis

´gamma
	µbeta
¶moving_mean
·moving_variance
¸	variables
¹regularization_losses
ºtrainable_variables
»	keras_api
+ú&call_and_return_all_conditional_losses
û__call__"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_198", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_198", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 5, 5, 144]}}
ü	
¼kernel
	½bias
¾	variables
¿regularization_losses
Àtrainable_variables
Á	keras_api
+ü&call_and_return_all_conditional_losses
ý__call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv2d_205", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_205", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 5, 5, 144]}}
Ä	
	Âaxis

Ãgamma
	Äbeta
Åmoving_mean
Æmoving_variance
Ç	variables
Èregularization_losses
Étrainable_variables
Ê	keras_api
+þ&call_and_return_all_conditional_losses
ÿ__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_199", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_199", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 3, 3, 48]}}

Ë	variables
Ìregularization_losses
Ítrainable_variables
Î	keras_api
+&call_and_return_all_conditional_losses
__call__"ò
_tf_keras_layerØ{"class_name": "MaxPooling2D", "name": "max_pooling2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ý	
Ïkernel
	Ðbias
Ñ	variables
Òregularization_losses
Ótrainable_variables
Ô	keras_api
+&call_and_return_all_conditional_losses
__call__"Ð
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv2d_206", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_206", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 3, 3, 192]}}
Æ	
	Õaxis

Ögamma
	×beta
Ømoving_mean
Ùmoving_variance
Ú	variables
Ûregularization_losses
Ütrainable_variables
Ý	keras_api
+&call_and_return_all_conditional_losses
__call__"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_200", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_200", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 3, 3, 144]}}
ü	
Þkernel
	ßbias
à	variables
áregularization_losses
âtrainable_variables
ã	keras_api
+&call_and_return_all_conditional_losses
__call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv2d_207", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_207", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 3, 3, 144]}}
Ä	
	äaxis

ågamma
	æbeta
çmoving_mean
èmoving_variance
é	variables
êregularization_losses
ëtrainable_variables
ì	keras_api
+&call_and_return_all_conditional_losses
__call__"å
_tf_keras_layerË{"class_name": "BatchNormalization", "name": "batch_normalization_201", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_201", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 3, 3, 48]}}
ì
í	variables
îregularization_losses
ïtrainable_variables
ð	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ú
ñkernel
	òbias
ó	variables
ôregularization_losses
õtrainable_variables
ö	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1728}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 1728]}}
÷
÷kernel
	øbias
ù	variables
úregularization_losses
ûtrainable_variables
ü	keras_api
+&call_and_return_all_conditional_losses
__call__"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 256]}}
÷
ýkernel
	þbias
ÿ	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 64]}}
ú
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1728}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 1728]}}
ú
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 256]}}
ú
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 256]}}
é
)0
*1
02
13
24
35
86
97
?8
@9
A10
B11
G12
H13
N14
O15
P16
Q17
V18
W19
]20
^21
_22
`23
i24
j25
p26
q27
r28
s29
x30
y31
32
33
34
35
36
37
38
39
40
41
42
43
¡44
¢45
£46
¤47
­48
®49
´50
µ51
¶52
·53
¼54
½55
Ã56
Ä57
Å58
Æ59
Ï60
Ð61
Ö62
×63
Ø64
Ù65
Þ66
ß67
å68
æ69
ç70
è71
ñ72
ò73
÷74
ø75
ý76
þ77
78
79
80
81
82
83"
trackable_list_wrapper
 "
trackable_list_wrapper

)0
*1
02
13
84
95
?6
@7
G8
H9
N10
O11
V12
W13
]14
^15
i16
j17
p18
q19
x20
y21
22
23
24
25
26
27
28
29
¡30
¢31
­32
®33
´34
µ35
¼36
½37
Ã38
Ä39
Ï40
Ð41
Ö42
×43
Þ44
ß45
å46
æ47
ñ48
ò49
÷50
ø51
ý52
þ53
54
55
56
57
58
59"
trackable_list_wrapper
Ó
non_trainable_variables
$	variables
layers
%regularization_losses
 layer_regularization_losses
layer_metrics
&trainable_variables
metrics
Ê__call__
Ë_default_save_signature
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
4:22rcnn_1/conv2d_196/kernel
%:#2rcnn_1/conv2d_196/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
µ
non_trainable_variables
+	variables
layers
,regularization_losses
 layer_regularization_losses
layer_metrics
-trainable_variables
metrics
Ó__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
3:12$rcnn_1/batch_normalization_190/gamma
2:02#rcnn_1/batch_normalization_190/beta
;:9 (2*rcnn_1/batch_normalization_190/moving_mean
?:= (2.rcnn_1/batch_normalization_190/moving_variance
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
µ
non_trainable_variables
4	variables
 layers
5regularization_losses
 ¡layer_regularization_losses
¢layer_metrics
6trainable_variables
£metrics
Õ__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
4:2À2rcnn_1/conv2d_197/kernel
%:#À2rcnn_1/conv2d_197/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
¤non_trainable_variables
:	variables
¥layers
;regularization_losses
 ¦layer_regularization_losses
§layer_metrics
<trainable_variables
¨metrics
×__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
3:1À2$rcnn_1/batch_normalization_191/gamma
2:0À2#rcnn_1/batch_normalization_191/beta
;:9À (2*rcnn_1/batch_normalization_191/moving_mean
?:=À (2.rcnn_1/batch_normalization_191/moving_variance
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
µ
©non_trainable_variables
C	variables
ªlayers
Dregularization_losses
 «layer_regularization_losses
¬layer_metrics
Etrainable_variables
­metrics
Ù__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
4:2À2rcnn_1/conv2d_198/kernel
%:#2rcnn_1/conv2d_198/bias
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
µ
®non_trainable_variables
I	variables
¯layers
Jregularization_losses
 °layer_regularization_losses
±layer_metrics
Ktrainable_variables
²metrics
Û__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
3:12$rcnn_1/batch_normalization_192/gamma
2:02#rcnn_1/batch_normalization_192/beta
;:9 (2*rcnn_1/batch_normalization_192/moving_mean
?:= (2.rcnn_1/batch_normalization_192/moving_variance
<
N0
O1
P2
Q3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
µ
³non_trainable_variables
R	variables
´layers
Sregularization_losses
 µlayer_regularization_losses
¶layer_metrics
Ttrainable_variables
·metrics
Ý__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
3:1		02rcnn_1/conv2d_199/kernel
$:"02rcnn_1/conv2d_199/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
µ
¸non_trainable_variables
X	variables
¹layers
Yregularization_losses
 ºlayer_regularization_losses
»layer_metrics
Ztrainable_variables
¼metrics
ß__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
2:002$rcnn_1/batch_normalization_193/gamma
1:/02#rcnn_1/batch_normalization_193/beta
::80 (2*rcnn_1/batch_normalization_193/moving_mean
>:<0 (2.rcnn_1/batch_normalization_193/moving_variance
<
]0
^1
_2
`3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
µ
½non_trainable_variables
a	variables
¾layers
bregularization_losses
 ¿layer_regularization_losses
Àlayer_metrics
ctrainable_variables
Ámetrics
á__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ânon_trainable_variables
e	variables
Ãlayers
fregularization_losses
 Älayer_regularization_losses
Ålayer_metrics
gtrainable_variables
Æmetrics
ã__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
4:2À2rcnn_1/conv2d_200/kernel
%:#2rcnn_1/conv2d_200/bias
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
µ
Çnon_trainable_variables
k	variables
Èlayers
lregularization_losses
 Élayer_regularization_losses
Êlayer_metrics
mtrainable_variables
Ëmetrics
å__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
3:12$rcnn_1/batch_normalization_194/gamma
2:02#rcnn_1/batch_normalization_194/beta
;:9 (2*rcnn_1/batch_normalization_194/moving_mean
?:= (2.rcnn_1/batch_normalization_194/moving_variance
<
p0
q1
r2
s3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
µ
Ìnon_trainable_variables
t	variables
Ílayers
uregularization_losses
 Îlayer_regularization_losses
Ïlayer_metrics
vtrainable_variables
Ðmetrics
ç__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
3:102rcnn_1/conv2d_201/kernel
$:"02rcnn_1/conv2d_201/bias
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
µ
Ñnon_trainable_variables
z	variables
Òlayers
{regularization_losses
 Ólayer_regularization_losses
Ôlayer_metrics
|trainable_variables
Õmetrics
é__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
2:002$rcnn_1/batch_normalization_195/gamma
1:/02#rcnn_1/batch_normalization_195/beta
::80 (2*rcnn_1/batch_normalization_195/moving_mean
>:<0 (2.rcnn_1/batch_normalization_195/moving_variance
?
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
¸
Önon_trainable_variables
	variables
×layers
regularization_losses
 Ølayer_regularization_losses
Ùlayer_metrics
trainable_variables
Úmetrics
ë__call__
+ê&call_and_return_all_conditional_losses
'ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûnon_trainable_variables
	variables
Ülayers
regularization_losses
 Ýlayer_regularization_losses
Þlayer_metrics
trainable_variables
ßmetrics
í__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
4:2À2rcnn_1/conv2d_202/kernel
%:#2rcnn_1/conv2d_202/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ànon_trainable_variables
	variables
álayers
regularization_losses
 âlayer_regularization_losses
ãlayer_metrics
trainable_variables
ämetrics
ï__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
3:12$rcnn_1/batch_normalization_196/gamma
2:02#rcnn_1/batch_normalization_196/beta
;:9 (2*rcnn_1/batch_normalization_196/moving_mean
?:= (2.rcnn_1/batch_normalization_196/moving_variance
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ånon_trainable_variables
	variables
ælayers
regularization_losses
 çlayer_regularization_losses
èlayer_metrics
trainable_variables
émetrics
ñ__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
3:102rcnn_1/conv2d_203/kernel
$:"02rcnn_1/conv2d_203/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ênon_trainable_variables
	variables
ëlayers
regularization_losses
 ìlayer_regularization_losses
ílayer_metrics
trainable_variables
îmetrics
ó__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
2:002$rcnn_1/batch_normalization_197/gamma
1:/02#rcnn_1/batch_normalization_197/beta
::80 (2*rcnn_1/batch_normalization_197/moving_mean
>:<0 (2.rcnn_1/batch_normalization_197/moving_variance
@
¡0
¢1
£2
¤3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
¸
ïnon_trainable_variables
¥	variables
ðlayers
¦regularization_losses
 ñlayer_regularization_losses
òlayer_metrics
§trainable_variables
ómetrics
õ__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ônon_trainable_variables
©	variables
õlayers
ªregularization_losses
 ölayer_regularization_losses
÷layer_metrics
«trainable_variables
ømetrics
÷__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
4:2À2rcnn_1/conv2d_204/kernel
%:#2rcnn_1/conv2d_204/bias
0
­0
®1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
­0
®1"
trackable_list_wrapper
¸
ùnon_trainable_variables
¯	variables
úlayers
°regularization_losses
 ûlayer_regularization_losses
ülayer_metrics
±trainable_variables
ýmetrics
ù__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
3:12$rcnn_1/batch_normalization_198/gamma
2:02#rcnn_1/batch_normalization_198/beta
;:9 (2*rcnn_1/batch_normalization_198/moving_mean
?:= (2.rcnn_1/batch_normalization_198/moving_variance
@
´0
µ1
¶2
·3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
´0
µ1"
trackable_list_wrapper
¸
þnon_trainable_variables
¸	variables
ÿlayers
¹regularization_losses
 layer_regularization_losses
layer_metrics
ºtrainable_variables
metrics
û__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
3:102rcnn_1/conv2d_205/kernel
$:"02rcnn_1/conv2d_205/bias
0
¼0
½1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
¼0
½1"
trackable_list_wrapper
¸
non_trainable_variables
¾	variables
layers
¿regularization_losses
 layer_regularization_losses
layer_metrics
Àtrainable_variables
metrics
ý__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
2:002$rcnn_1/batch_normalization_199/gamma
1:/02#rcnn_1/batch_normalization_199/beta
::80 (2*rcnn_1/batch_normalization_199/moving_mean
>:<0 (2.rcnn_1/batch_normalization_199/moving_variance
@
Ã0
Ä1
Å2
Æ3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
¸
non_trainable_variables
Ç	variables
layers
Èregularization_losses
 layer_regularization_losses
layer_metrics
Étrainable_variables
metrics
ÿ__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
Ë	variables
layers
Ìregularization_losses
 layer_regularization_losses
layer_metrics
Ítrainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
4:2À2rcnn_1/conv2d_206/kernel
%:#2rcnn_1/conv2d_206/bias
0
Ï0
Ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
¸
non_trainable_variables
Ñ	variables
layers
Òregularization_losses
 layer_regularization_losses
layer_metrics
Ótrainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
3:12$rcnn_1/batch_normalization_200/gamma
2:02#rcnn_1/batch_normalization_200/beta
;:9 (2*rcnn_1/batch_normalization_200/moving_mean
?:= (2.rcnn_1/batch_normalization_200/moving_variance
@
Ö0
×1
Ø2
Ù3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ö0
×1"
trackable_list_wrapper
¸
non_trainable_variables
Ú	variables
layers
Ûregularization_losses
 layer_regularization_losses
layer_metrics
Ütrainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
3:102rcnn_1/conv2d_207/kernel
$:"02rcnn_1/conv2d_207/bias
0
Þ0
ß1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Þ0
ß1"
trackable_list_wrapper
¸
non_trainable_variables
à	variables
layers
áregularization_losses
 layer_regularization_losses
layer_metrics
âtrainable_variables
 metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
2:002$rcnn_1/batch_normalization_201/gamma
1:/02#rcnn_1/batch_normalization_201/beta
::80 (2*rcnn_1/batch_normalization_201/moving_mean
>:<0 (2.rcnn_1/batch_normalization_201/moving_variance
@
å0
æ1
ç2
è3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
å0
æ1"
trackable_list_wrapper
¸
¡non_trainable_variables
é	variables
¢layers
êregularization_losses
 £layer_regularization_losses
¤layer_metrics
ëtrainable_variables
¥metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¦non_trainable_variables
í	variables
§layers
îregularization_losses
 ¨layer_regularization_losses
©layer_metrics
ïtrainable_variables
ªmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'
À2rcnn_1/dense_6/kernel
": 2rcnn_1/dense_6/bias
0
ñ0
ò1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ñ0
ò1"
trackable_list_wrapper
¸
«non_trainable_variables
ó	variables
¬layers
ôregularization_losses
 ­layer_regularization_losses
®layer_metrics
õtrainable_variables
¯metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
(:&	@2rcnn_1/dense_7/kernel
!:@2rcnn_1/dense_7/bias
0
÷0
ø1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
÷0
ø1"
trackable_list_wrapper
¸
°non_trainable_variables
ù	variables
±layers
úregularization_losses
 ²layer_regularization_losses
³layer_metrics
ûtrainable_variables
´metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%@2rcnn_1/dense_8/kernel
!:2rcnn_1/dense_8/bias
0
ý0
þ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ý0
þ1"
trackable_list_wrapper
¸
µnon_trainable_variables
ÿ	variables
¶layers
regularization_losses
 ·layer_regularization_losses
¸layer_metrics
trainable_variables
¹metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'
À2rcnn_1/dense_9/kernel
": 2rcnn_1/dense_9/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ºnon_trainable_variables
	variables
»layers
regularization_losses
 ¼layer_regularization_losses
½layer_metrics
trainable_variables
¾metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(
2rcnn_1/dense_10/kernel
#:!2rcnn_1/dense_10/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
¿non_trainable_variables
	variables
Àlayers
regularization_losses
 Álayer_regularization_losses
Âlayer_metrics
trainable_variables
Ãmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'	2rcnn_1/dense_11/kernel
": 2rcnn_1/dense_11/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
Änon_trainable_variables
	variables
Ålayers
regularization_losses
 Ælayer_regularization_losses
Çlayer_metrics
trainable_variables
Èmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ä
20
31
A2
B3
P4
Q5
_6
`7
r8
s9
10
11
12
13
£14
¤15
¶16
·17
Å18
Æ19
Ø20
Ù21
ç22
è23"
trackable_list_wrapper
®
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
£0
¤1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
¶0
·1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ø0
Ù1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
ç0
è1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Æ2Ã
A__inference_rcnn_1_layer_call_and_return_conditional_losses_63347
A__inference_rcnn_1_layer_call_and_return_conditional_losses_62531
A__inference_rcnn_1_layer_call_and_return_conditional_losses_61269
A__inference_rcnn_1_layer_call_and_return_conditional_losses_60453´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
&__inference_rcnn_1_layer_call_fn_61691
&__inference_rcnn_1_layer_call_fn_63769
&__inference_rcnn_1_layer_call_fn_63558
&__inference_rcnn_1_layer_call_fn_61480´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
 __inference__wrapped_model_53817°
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢

*'
input_1ÿÿÿÿÿÿÿÿÿ
åá
+(
	input_2_1ÿÿÿÿÿÿÿÿÿ
+(
	input_2_2ÿÿÿÿÿÿÿÿÿ
+(
	input_2_3ÿÿÿÿÿÿÿÿÿ
+(
	input_2_4ÿÿÿÿÿÿÿÿÿ
+(
	input_2_5ÿÿÿÿÿÿÿÿÿ
è2å
__inference_getROIfeature_53038Á
¡²
FullArgSpec%
args
jself
jinputs
jsize
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
 
Ù2Ö
!__inference_getROIfeature12_53559°
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
Ø2Õ
 __inference_getROIfeature2_53046°
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
Ø2Õ
 __inference_getROIfeature3_53175°
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
Ø2Õ
 __inference_getROIfeature5_53303°
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
Ø2Õ
 __inference_getROIfeature8_53431°
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ï2ì
E__inference_conv2d_196_layer_call_and_return_conditional_losses_63780¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_196_layer_call_fn_63789¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63827
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63891
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63873
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63809´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_190_layer_call_fn_63904
7__inference_batch_normalization_190_layer_call_fn_63840
7__inference_batch_normalization_190_layer_call_fn_63917
7__inference_batch_normalization_190_layer_call_fn_63853´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_197_layer_call_and_return_conditional_losses_63928¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_197_layer_call_fn_63937¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_63957
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_64039
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_64021
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_63975´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_191_layer_call_fn_64052
7__inference_batch_normalization_191_layer_call_fn_64001
7__inference_batch_normalization_191_layer_call_fn_63988
7__inference_batch_normalization_191_layer_call_fn_64065´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_198_layer_call_and_return_conditional_losses_64076¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_198_layer_call_fn_64085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64105
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64123
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64169
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64187´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_192_layer_call_fn_64200
7__inference_batch_normalization_192_layer_call_fn_64149
7__inference_batch_normalization_192_layer_call_fn_64213
7__inference_batch_normalization_192_layer_call_fn_64136´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_199_layer_call_and_return_conditional_losses_64224¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_199_layer_call_fn_64233¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64253
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64271
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64317
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64335´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_193_layer_call_fn_64297
7__inference_batch_normalization_193_layer_call_fn_64348
7__inference_batch_normalization_193_layer_call_fn_64284
7__inference_batch_normalization_193_layer_call_fn_64361´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_54239à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_max_pooling2d_8_layer_call_fn_54245à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv2d_200_layer_call_and_return_conditional_losses_64372¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_200_layer_call_fn_64381¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64465
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64401
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64483
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64419´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_194_layer_call_fn_64445
7__inference_batch_normalization_194_layer_call_fn_64496
7__inference_batch_normalization_194_layer_call_fn_64432
7__inference_batch_normalization_194_layer_call_fn_64509´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_201_layer_call_and_return_conditional_losses_64520¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_201_layer_call_fn_64529¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64567
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64613
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64631
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64549´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_195_layer_call_fn_64657
7__inference_batch_normalization_195_layer_call_fn_64580
7__inference_batch_normalization_195_layer_call_fn_64593
7__inference_batch_normalization_195_layer_call_fn_64644´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_54459à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_max_pooling2d_9_layer_call_fn_54465à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv2d_202_layer_call_and_return_conditional_losses_64668¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_202_layer_call_fn_64677¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64715
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64697
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64761
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64779´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_196_layer_call_fn_64728
7__inference_batch_normalization_196_layer_call_fn_64741
7__inference_batch_normalization_196_layer_call_fn_64792
7__inference_batch_normalization_196_layer_call_fn_64805´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_203_layer_call_and_return_conditional_losses_64816¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_203_layer_call_fn_64825¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64909
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64845
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64927
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64863´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_197_layer_call_fn_64876
7__inference_batch_normalization_197_layer_call_fn_64889
7__inference_batch_normalization_197_layer_call_fn_64940
7__inference_batch_normalization_197_layer_call_fn_64953´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
³2°
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_54679à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
0__inference_max_pooling2d_10_layer_call_fn_54685à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv2d_204_layer_call_and_return_conditional_losses_64964¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_204_layer_call_fn_64973¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65075
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65011
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_64993
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65057´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_198_layer_call_fn_65088
7__inference_batch_normalization_198_layer_call_fn_65024
7__inference_batch_normalization_198_layer_call_fn_65101
7__inference_batch_normalization_198_layer_call_fn_65037´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_205_layer_call_and_return_conditional_losses_65112¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_205_layer_call_fn_65121¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65223
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65141
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65205
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65159´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_199_layer_call_fn_65185
7__inference_batch_normalization_199_layer_call_fn_65236
7__inference_batch_normalization_199_layer_call_fn_65172
7__inference_batch_normalization_199_layer_call_fn_65249´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
³2°
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_54899à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
0__inference_max_pooling2d_11_layer_call_fn_54905à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv2d_206_layer_call_and_return_conditional_losses_65260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_206_layer_call_fn_65269¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65307
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65289
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65371
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65353´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_200_layer_call_fn_65384
7__inference_batch_normalization_200_layer_call_fn_65397
7__inference_batch_normalization_200_layer_call_fn_65333
7__inference_batch_normalization_200_layer_call_fn_65320´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
E__inference_conv2d_207_layer_call_and_return_conditional_losses_65408¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_207_layer_call_fn_65417¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65437
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65501
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65455
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65519´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_201_layer_call_fn_65532
7__inference_batch_normalization_201_layer_call_fn_65468
7__inference_batch_normalization_201_layer_call_fn_65481
7__inference_batch_normalization_201_layer_call_fn_65545´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_flatten_1_layer_call_and_return_conditional_losses_65551¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_flatten_1_layer_call_fn_65556¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_6_layer_call_and_return_conditional_losses_65567¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_6_layer_call_fn_65576¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_7_layer_call_and_return_conditional_losses_65587¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_7_layer_call_fn_65596¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_8_layer_call_and_return_conditional_losses_65607¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_8_layer_call_fn_65616¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_9_layer_call_and_return_conditional_losses_65627¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_9_layer_call_fn_65636¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_10_layer_call_and_return_conditional_losses_65647¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_10_layer_call_fn_65656¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_11_layer_call_and_return_conditional_losses_65666¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_11_layer_call_fn_65675¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
gBe
#__inference_signature_wrapper_59613input_1	input_2_1	input_2_2	input_2_3	input_2_4	input_2_5
	J
Const
J	
Const_1ô	
 __inference__wrapped_model_53817Ï	)*012389?@ABÏÐÖ×ØÙÞßåæçè­®´µ¶·¼½ÃÄÅÆ¡¢£¤ijpqrsxyGHNOPQVW]^_`ñò÷øýþ«¢§
¢

*'
input_1ÿÿÿÿÿÿÿÿÿ
åá
+(
	input_2_1ÿÿÿÿÿÿÿÿÿ
+(
	input_2_2ÿÿÿÿÿÿÿÿÿ
+(
	input_2_3ÿÿÿÿÿÿÿÿÿ
+(
	input_2_4ÿÿÿÿÿÿÿÿÿ
+(
	input_2_5ÿÿÿÿÿÿÿÿÿ
ª "ª
2

output_1_1$!

output_1_1ÿÿÿÿÿÿÿÿÿ
2

output_1_2$!

output_1_2ÿÿÿÿÿÿÿÿÿ
2

output_1_3$!

output_1_3ÿÿÿÿÿÿÿÿÿ	
2

output_2_1$!

output_2_1ÿÿÿÿÿÿÿÿÿ
2

output_2_2$!

output_2_2ÿÿÿÿÿÿÿÿÿ
2

output_2_3$!

output_2_3ÿÿÿÿÿÿÿÿÿ	
2

output_3_1$!

output_3_1ÿÿÿÿÿÿÿÿÿ
2

output_3_2$!

output_3_2ÿÿÿÿÿÿÿÿÿ
2

output_3_3$!

output_3_3ÿÿÿÿÿÿÿÿÿ	
2

output_4_1$!

output_4_1ÿÿÿÿÿÿÿÿÿ
2

output_4_2$!

output_4_2ÿÿÿÿÿÿÿÿÿ
2

output_4_3$!

output_4_3ÿÿÿÿÿÿÿÿÿ	
2

output_5_1$!

output_5_1ÿÿÿÿÿÿÿÿÿ
2

output_5_2$!

output_5_2ÿÿÿÿÿÿÿÿÿ
2

output_5_3$!

output_5_3ÿÿÿÿÿÿÿÿÿ	ï
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_638090123N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_638270123N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63873t0123<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_190_layer_call_and_return_conditional_losses_63891t0123<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ç
7__inference_batch_normalization_190_layer_call_fn_638400123N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
7__inference_batch_normalization_190_layer_call_fn_638530123N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_190_layer_call_fn_63904g0123<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_190_layer_call_fn_63917g0123<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿÊ
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_63957t?@AB<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÀ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÀ
 Ê
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_63975t?@AB<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÀ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÀ
 ï
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_64021?@ABN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 ï
R__inference_batch_normalization_191_layer_call_and_return_conditional_losses_64039?@ABN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
 ¢
7__inference_batch_normalization_191_layer_call_fn_63988g?@AB<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÀ
p
ª "!ÿÿÿÿÿÿÿÿÿÀ¢
7__inference_batch_normalization_191_layer_call_fn_64001g?@AB<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿÀ
p 
ª "!ÿÿÿÿÿÿÿÿÿÀÇ
7__inference_batch_normalization_191_layer_call_fn_64052?@ABN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀÇ
7__inference_batch_normalization_191_layer_call_fn_64065?@ABN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀÊ
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64105tNOPQ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64123tNOPQ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ï
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64169NOPQN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
R__inference_batch_normalization_192_layer_call_and_return_conditional_losses_64187NOPQN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¢
7__inference_batch_normalization_192_layer_call_fn_64136gNOPQ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_192_layer_call_fn_64149gNOPQ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿÇ
7__inference_batch_normalization_192_layer_call_fn_64200NOPQN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
7__inference_batch_normalization_192_layer_call_fn_64213NOPQN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64253]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 í
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64271]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 È
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64317r]^_`;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 È
R__inference_batch_normalization_193_layer_call_and_return_conditional_losses_64335r]^_`;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 Å
7__inference_batch_normalization_193_layer_call_fn_64284]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0Å
7__inference_batch_normalization_193_layer_call_fn_64297]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0 
7__inference_batch_normalization_193_layer_call_fn_64348e]^_`;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª " ÿÿÿÿÿÿÿÿÿ0 
7__inference_batch_normalization_193_layer_call_fn_64361e]^_`;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª " ÿÿÿÿÿÿÿÿÿ0ï
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64401pqrsN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64419pqrsN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64465tpqrs<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_194_layer_call_and_return_conditional_losses_64483tpqrs<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ç
7__inference_batch_normalization_194_layer_call_fn_64432pqrsN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
7__inference_batch_normalization_194_layer_call_fn_64445pqrsN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_194_layer_call_fn_64496gpqrs<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_194_layer_call_fn_64509gpqrs<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿË
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64549u;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 Ë
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64567u;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 ð
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64613M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 ð
R__inference_batch_normalization_195_layer_call_and_return_conditional_losses_64631M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 £
7__inference_batch_normalization_195_layer_call_fn_64580h;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª " ÿÿÿÿÿÿÿÿÿ0£
7__inference_batch_normalization_195_layer_call_fn_64593h;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª " ÿÿÿÿÿÿÿÿÿ0È
7__inference_batch_normalization_195_layer_call_fn_64644M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0È
7__inference_batch_normalization_195_layer_call_fn_64657M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0ó
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64697N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ó
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64715N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Î
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64761x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Î
R__inference_batch_normalization_196_layer_call_and_return_conditional_losses_64779x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ë
7__inference_batch_normalization_196_layer_call_fn_64728N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
7__inference_batch_normalization_196_layer_call_fn_64741N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
7__inference_batch_normalization_196_layer_call_fn_64792k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¦
7__inference_batch_normalization_196_layer_call_fn_64805k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿñ
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64845¡¢£¤M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 ñ
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64863¡¢£¤M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 Ì
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64909v¡¢£¤;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 Ì
R__inference_batch_normalization_197_layer_call_and_return_conditional_losses_64927v¡¢£¤;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 É
7__inference_batch_normalization_197_layer_call_fn_64876¡¢£¤M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0É
7__inference_batch_normalization_197_layer_call_fn_64889¡¢£¤M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0¤
7__inference_batch_normalization_197_layer_call_fn_64940i¡¢£¤;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª " ÿÿÿÿÿÿÿÿÿ0¤
7__inference_batch_normalization_197_layer_call_fn_64953i¡¢£¤;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª " ÿÿÿÿÿÿÿÿÿ0ó
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_64993´µ¶·N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ó
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65011´µ¶·N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Î
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65057x´µ¶·<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Î
R__inference_batch_normalization_198_layer_call_and_return_conditional_losses_65075x´µ¶·<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ë
7__inference_batch_normalization_198_layer_call_fn_65024´µ¶·N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
7__inference_batch_normalization_198_layer_call_fn_65037´µ¶·N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
7__inference_batch_normalization_198_layer_call_fn_65088k´µ¶·<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¦
7__inference_batch_normalization_198_layer_call_fn_65101k´µ¶·<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿñ
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65141ÃÄÅÆM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 ñ
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65159ÃÄÅÆM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 Ì
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65205vÃÄÅÆ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 Ì
R__inference_batch_normalization_199_layer_call_and_return_conditional_losses_65223vÃÄÅÆ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 É
7__inference_batch_normalization_199_layer_call_fn_65172ÃÄÅÆM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0É
7__inference_batch_normalization_199_layer_call_fn_65185ÃÄÅÆM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0¤
7__inference_batch_normalization_199_layer_call_fn_65236iÃÄÅÆ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª " ÿÿÿÿÿÿÿÿÿ0¤
7__inference_batch_normalization_199_layer_call_fn_65249iÃÄÅÆ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª " ÿÿÿÿÿÿÿÿÿ0ó
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65289Ö×ØÙN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ó
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65307Ö×ØÙN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Î
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65353xÖ×ØÙ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Î
R__inference_batch_normalization_200_layer_call_and_return_conditional_losses_65371xÖ×ØÙ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ë
7__inference_batch_normalization_200_layer_call_fn_65320Ö×ØÙN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
7__inference_batch_normalization_200_layer_call_fn_65333Ö×ØÙN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
7__inference_batch_normalization_200_layer_call_fn_65384kÖ×ØÙ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¦
7__inference_batch_normalization_200_layer_call_fn_65397kÖ×ØÙ<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿñ
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65437åæçèM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 ñ
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65455åæçèM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 Ì
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65501våæçè;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 Ì
R__inference_batch_normalization_201_layer_call_and_return_conditional_losses_65519våæçè;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 É
7__inference_batch_normalization_201_layer_call_fn_65468åæçèM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0É
7__inference_batch_normalization_201_layer_call_fn_65481åæçèM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0¤
7__inference_batch_normalization_201_layer_call_fn_65532iåæçè;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p
ª " ÿÿÿÿÿÿÿÿÿ0¤
7__inference_batch_normalization_201_layer_call_fn_65545iåæçè;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0
p 
ª " ÿÿÿÿÿÿÿÿÿ0·
E__inference_conv2d_196_layer_call_and_return_conditional_losses_63780n)*8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_196_layer_call_fn_63789a)*8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ·
E__inference_conv2d_197_layer_call_and_return_conditional_losses_63928n898¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÀ
 
*__inference_conv2d_197_layer_call_fn_63937a898¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÀ·
E__inference_conv2d_198_layer_call_and_return_conditional_losses_64076nGH8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_198_layer_call_fn_64085aGH8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª "!ÿÿÿÿÿÿÿÿÿ¶
E__inference_conv2d_199_layer_call_and_return_conditional_losses_64224mVW8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 
*__inference_conv2d_199_layer_call_fn_64233`VW8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ0·
E__inference_conv2d_200_layer_call_and_return_conditional_losses_64372nij8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_200_layer_call_fn_64381aij8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª "!ÿÿÿÿÿÿÿÿÿ¶
E__inference_conv2d_201_layer_call_and_return_conditional_losses_64520mxy8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 
*__inference_conv2d_201_layer_call_fn_64529`xy8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ0¹
E__inference_conv2d_202_layer_call_and_return_conditional_losses_64668p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_202_layer_call_fn_64677c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª "!ÿÿÿÿÿÿÿÿÿ¸
E__inference_conv2d_203_layer_call_and_return_conditional_losses_64816o8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 
*__inference_conv2d_203_layer_call_fn_64825b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ0¹
E__inference_conv2d_204_layer_call_and_return_conditional_losses_64964p­®8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_204_layer_call_fn_64973c­®8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª "!ÿÿÿÿÿÿÿÿÿ¸
E__inference_conv2d_205_layer_call_and_return_conditional_losses_65112o¼½8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 
*__inference_conv2d_205_layer_call_fn_65121b¼½8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ0¹
E__inference_conv2d_206_layer_call_and_return_conditional_losses_65260pÏÐ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_206_layer_call_fn_65269cÏÐ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª "!ÿÿÿÿÿÿÿÿÿ¸
E__inference_conv2d_207_layer_call_and_return_conditional_losses_65408oÞß8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 
*__inference_conv2d_207_layer_call_fn_65417bÞß8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ0§
C__inference_dense_10_layer_call_and_return_conditional_losses_65647`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_dense_10_layer_call_fn_65656S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
C__inference_dense_11_layer_call_and_return_conditional_losses_65666_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
(__inference_dense_11_layer_call_fn_65675R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
B__inference_dense_6_layer_call_and_return_conditional_losses_65567`ñò0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
'__inference_dense_6_layer_call_fn_65576Sñò0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿ¥
B__inference_dense_7_layer_call_and_return_conditional_losses_65587_÷ø0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
'__inference_dense_7_layer_call_fn_65596R÷ø0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¤
B__inference_dense_8_layer_call_and_return_conditional_losses_65607^ýþ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_8_layer_call_fn_65616Qýþ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
B__inference_dense_9_layer_call_and_return_conditional_losses_65627`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
'__inference_dense_9_layer_call_fn_65636S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿª
D__inference_flatten_1_layer_call_and_return_conditional_losses_65551b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 
)__inference_flatten_1_layer_call_fn_65556U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿÀî
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_54679R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_10_layer_call_fn_54685R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_54899R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_11_layer_call_fn_54905R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_54239R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_8_layer_call_fn_54245R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_54459R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_9_layer_call_fn_54465R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ

#__inference_signature_wrapper_59613
)*012389?@ABÏÐÖ×ØÙÞßåæçè­®´µ¶·¼½ÃÄÅÆ¡¢£¤ijpqrsxyGHNOPQVW]^_`ñò÷øýþé¢å
¢ 
ÝªÙ
5
input_1*'
input_1ÿÿÿÿÿÿÿÿÿ
8
	input_2_1+(
	input_2_1ÿÿÿÿÿÿÿÿÿ
8
	input_2_2+(
	input_2_2ÿÿÿÿÿÿÿÿÿ
8
	input_2_3+(
	input_2_3ÿÿÿÿÿÿÿÿÿ
8
	input_2_4+(
	input_2_4ÿÿÿÿÿÿÿÿÿ
8
	input_2_5+(
	input_2_5ÿÿÿÿÿÿÿÿÿ"ª
2

output_1_1$!

output_1_1ÿÿÿÿÿÿÿÿÿ
2

output_1_2$!

output_1_2ÿÿÿÿÿÿÿÿÿ
2

output_1_3$!

output_1_3ÿÿÿÿÿÿÿÿÿ	
2

output_2_1$!

output_2_1ÿÿÿÿÿÿÿÿÿ
2

output_2_2$!

output_2_2ÿÿÿÿÿÿÿÿÿ
2

output_2_3$!

output_2_3ÿÿÿÿÿÿÿÿÿ	
2

output_3_1$!

output_3_1ÿÿÿÿÿÿÿÿÿ
2

output_3_2$!

output_3_2ÿÿÿÿÿÿÿÿÿ
2

output_3_3$!

output_3_3ÿÿÿÿÿÿÿÿÿ	
2

output_4_1$!

output_4_1ÿÿÿÿÿÿÿÿÿ
2

output_4_2$!

output_4_2ÿÿÿÿÿÿÿÿÿ
2

output_4_3$!

output_4_3ÿÿÿÿÿÿÿÿÿ	
2

output_5_1$!

output_5_1ÿÿÿÿÿÿÿÿÿ
2

output_5_2$!

output_5_2ÿÿÿÿÿÿÿÿÿ
2

output_5_3$!

output_5_3ÿÿÿÿÿÿÿÿÿ	