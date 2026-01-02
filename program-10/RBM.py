# Restricted Boltzmann Machine（RBM）
import numpy as np
import sys
import random
from PIL import Image
import matplotlib.pyplot as plt

# 画像の大きさ
X_SIZE = 14
Y_SIZE = 14

# 可視層の個数
V = X_SIZE * Y_SIZE
    
# 隠れ層の個数
H = 64

# クラス数
class_num = 10

# 学習（テスト）データ数
train_num = 100

# 画像データ
img = np.zeros((class_num,train_num,V), dtype=np.float64)

# データの読み込み
def Read_data( flag ):
	dir = [ "train" , "test" ]
	for i in range(class_num):
		for j in range(1,train_num+1):
			# グレースケール画像で読み込み→大きさの変更→numpyに変換，ベクトル化→二値化
			train_file = "mnist/" + dir[ flag ] + "/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
			work_img = Image.open(train_file).convert('L')
			resize_img = work_img.resize((Y_SIZE, X_SIZE))
			resize_img = np.asarray(resize_img).astype(np.float64).flatten()
			img[i][j-1] = np.where(resize_img/255 < 0.5, 0, 1)

# RBM
class RBM:
	def __init__(self, n_v, n_h):
		# 結合係数
		self.w = np.random.randn(n_h, n_v)
		
		# 隠れ層の閾値
		self.b = np.random.randn(1, n_h)
		
		# 可視層の閾値
		self.a = np.random.randn(1, n_v)

	# シグモイド関数
	def Sigmoid( self, x ):
		return 1.0 / (1.0 + np.exp(-x))

	# 可視層→隠れ層
	def Encode( self, v ):
		p_h = self.Sigmoid(v.dot(self.w.T) + self.b)
		h = (np.random.rand(1,H) < p_h).astype('float64')
		return h
	
	# 隠れ層→可視層
	def Decode( self, h ):
		p_v = self.Sigmoid(h.dot(self.w) + self.a)
		v = (np.random.rand(1,V) < p_v).astype('float64')
		return v
	
	# ギブスサンプリング
	def GibbsSampling( self, v ):
		T = 10
		for t in range(T):
			h = rbm.Encode( v )
			v = rbm.Decode( h )
		return v , h
	
	# Contrastive Divergence
	def Update(self, v_0, v_t, h_0, h_t ):
		alpha = 0.01
		self.w += alpha * (h_0.T.dot(v_0) - h_t.T.dot(v_t) )
		self.a += alpha * (v_0 - v_t)
		self.b += alpha * (h_0 - h_t)
	
	# パラメーターの保存
	def Save(self, filename):
		# 重み，閾値の保存
		np.savez(filename, w=self.w, a=self.a, b=self.b)
	
	# パラメータのロード
	def Load(self, filename):
		# 重み，閾値のロード
		work = np.load(filename)
		self.w = work['w']
		self.a = work['a']
		self.b = work['b']

# 学習
def Train():
	LOOP = 100
	for epoch in range(LOOP):
		if epoch % 10 == 0:
			print( epoch )
			
		for l in range( class_num * train_num ):
		
			# v_0 の選択
			c = random.randint( 0, class_num-1 )
			t = random.randint( 0, train_num-1 )
			v = img[c][t].copy()
			v_0 = v.reshape([1, V])
		
			# h_0 の計算
			h_0 = rbm.Encode( v_0 )
		
			# ギブスサンプリング
			v_t , h_t  = rbm.GibbsSampling( v_0 )
		
			# CD法による更新
			rbm.Update( v_0, v_t, h_0, h_t )
	
	# 重みの保存
	rbm.Save( "dat/rbm.npz" )

if __name__ == '__main__':
	
	# RBMの初期化
	rbm = RBM( V, H )
	
	argvs = sys.argv
	
	# 引数がtの場合
	if argvs[1] == "t":
	
		# 学習データの読み込み
		flag = 0
		Read_data( flag )
		
		# 学習
		Train()
		
	# 引数がpの場合
	elif argvs[1] == "p":
	
		# テストデータの読み込み
		flag = 1
		Read_data( flag )
		
		# 結合係数のロード
		rbm.Load( "dat/rbm.npz" )
		
		# 可視層(v_0)→隠れ層(h_0)→可視層(v_1)
		for i in range(10):
		
			# v_0 の選択
			c = random.randint( 0, class_num-1 )
			t = random.randint( 0, train_num-1 )
			v = img[c][t].copy()
			v_0 = v.reshape([1, V])
		
			# h_0 の計算
			h_0 = rbm.Encode( v_0 )
			
			# v_1の計算
			v_1 = rbm.Decode( h_0 )
		
			# 画像の描画
			plt.figure()
		
			# 元画像の表示
			plt.subplot(1,2,1)
			a = np.reshape( v_0[0] , (Y_SIZE,X_SIZE) )
			plt.imshow(a,cmap='gray')
			plt.title( "Original Image" )
		
			# 復元画像の表示
			plt.subplot(1,2,2)
			b = np.reshape( v_1[0] , (Y_SIZE,X_SIZE) )
			plt.imshow(b,cmap='gray')
		
			# 画像の保存
			plt.title( "Restore Image" )
			file = "dat/result-" + str(i) + ".png"
			plt.savefig(file)
			plt.close()
