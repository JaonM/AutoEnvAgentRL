"""New synthetic holdout dossiers. These are test inputs, never author prompts or patches."""
import argparse
import copy
import json
from pathlib import Path


def table(identity, rows): return {'id':identity,'rows':rows}
def generated(identity,count,columns): return {'id':identity,'generator':{'seed':0,'count':count,'columns':columns}}
def seq(start=0,step=1,**extra): return {'kind':'sequence','start':start,'step':step,**extra}
def constant(value): return {'kind':'constant','value':value}
def cycle(values): return {'kind':'cycle','values':values}


def cases():
    result=[]
    def add(family,mode,stratum,title,scene,policy,tables,facts,params,known=None):
        result.append({'id':len(result)+1,'business_family':family,'stratum':stratum,'title':title,
            'user_scene':{'situation_and_desired_outcome':scene,'known_information':known or {}},
            'business':{'provenance':'independently authored synthetic holdout dossier, not frozen',
                'policy':policy + (' 本数据集为单租户已认证工作区；调用者对所列记录具备查询和按上述规则操作的权限。该权限不替代每次业务操作要求的用户确认。' if stratum!='direct' else ''),'tables':tables},'user':{'facts':facts},
            'acceptance':{'kind':f'fresh_{family}_{mode}','params':params,'observations':{},'mutation':None}})
    parcel={'grams':750,'volume_grams':1250,'remote':True,'zone':'N','hazardous':False}
    service={'unit_cents':180,'remote_cents':250}
    tariff='计费重量为实重 grams 与体积折算重量 volume_grams 的较大值，每500克向上取整计一档，费用为档数乘 unit_cents，偏远地址另加 remote_cents；全部费用单位分。'
    add('parcel','direct','direct','体积重量与偏远附加费','想算这件包裹的运费，告诉我总共多少分。',tariff,[],{},dict(parcel=parcel,service=service),{'包裹':parcel,'计费标准':service})
    parcels=[dict(id='P1',**parcel,status='pending',service_id='',dispatch_id='D1'),
             dict(id='P2',**{**parcel,'grams':900,'volume_grams':900},status='pending',service_id='',dispatch_id='D2')]
    services=[{'id':'S1','zone':'N','remaining':1,'max_grams':2000,'hazardous':False,'unit_cents':100,'remote_cents':200},
              {'id':'S2','zone':'N','remaining':3,'max_grams':1000,'hazardous':True,'unit_cents':50,'remote_cents':100},
              {'id':'S3','zone':'N','remaining':0,'max_grams':5000,'hazardous':True,'unit_cents':1,'remote_cents':0}]
    base=[table('parcels',parcels),table('services',services)]
    policy=tariff+'服务仅在区域相同、剩余名额至少1、重量上限涵盖计费重量且危险品许可适用时可选；仅危险品需要 hazardous=true。同价按服务ID字典序选小者。'
    add('parcel','simple','simple','承运资格核验','帮我查一下我的包裹能不能走我指定的服务。',policy,copy.deepcopy(base),{'parcel_id':'P1','service_id':'S2'}, {})
    add('parcel','ordinary','ordinary','完整承运方案比较','帮我查全这件包裹能选的配送服务，并告诉我其中最低运费。',policy,copy.deepcopy(base),{'parcel_id':'P1'}, {})
    writes=policy+'按用户给的包裹顺序处理，每件选择当前可选且费用最低的服务，钱包必须足额；每件独立展示包裹、服务和费用并确认。确认后原子扣一个名额和运费，包裹状态改 sent 并记录 service_id，对应 dispatches.done=true、charge_cents=本件费用。只允许 status=pending 且对应 dispatches.done=false 的包裹办理，其他状态不得重复扣名额或费用。不可用、余额不足或拒绝时不改该件；已成功件保留。后件用前件完成后的真实状态重新选择。'
    extra=[table('wallets',[{'id':'W1','cents':550}]),table('dispatches',[{'id':'D1','done':False,'charge_cents':0},{'id':'D2','done':False,'charge_cents':0}])]
    add('parcel','partial','complex','动态承运名额与独立寄件','按顺序帮我寄这两件，每件先告诉我选哪家、多少钱再问我；办不成的别影响已经寄好的，最后告诉我寄出了哪些和钱包余额。',writes,copy.deepcopy(base+extra),{'parcel_ids':['P1','P2'],'wallet_id':'W1','approve_0':True,'approve_1':True},{})
    large=copy.deepcopy(base+extra);large[1]=generated('services',384,{'id':seq(0,prefix='S',width=4),'zone':constant('N'),'remaining':constant(2),'max_grams':constant(5000),'hazardous':constant(True),'unit_cents':seq(484,-1),'remote_cents':constant(0)})
    large[2]['rows'][0]['cents']=10000
    add('parcel','large','complex','大目录承运检索与原子寄件','请查全所有配送方案，帮这件包裹挑最便宜的；费用确认后再寄，最后告诉我寄出了哪些包裹和钱包余额。',writes+'本任务必须完整读取服务目录后比较，不可只比较第一页。',large,{'parcel_ids':['P1'],'wallet_id':'W1','approve_0':True},{})

    parking_price={'first_hour_cents':300,'half_hour_cents':120,'daily_cap_cents':1500}
    parking_policy='一次停车前60分钟收费 first_hour_cents，超过部分每30分钟向上取整收费 half_hour_cents，总费用不超过 daily_cap_cents，均为分。时间为当日分钟，所有区间均左闭右开；两区间仅端点相接不算重叠。'
    add('parking','direct','direct','阶梯停车计费与封顶','这次停了这些分钟，停车费是多少分？',parking_policy,[],{},dict(parking_price,minutes=181),dict(parking_price,minutes=181))
    bays=[{'id':'B1','active':True,'charger':True},{'id':'B2','active':True,'charger':False},{'id':'B3','active':False,'charger':True}]
    bookings=[{'id':'E1','bay_id':'B1','start':600,'end':660,'status':'booked','paid_cents':300},
              {'id':'N1','bay_id':'B1','start':660,'end':750,'status':'pending','paid_cents':0},
              {'id':'N2','bay_id':'B2','start':660,'end':750,'status':'pending','paid_cents':0}]
    tables=[table('bays',bays),table('bookings',bookings)]
    availability=parking_policy+'车位须 active=true；需要充电时还须 charger=true。只把 status=booked 的其他预约视为占用，pending 不占用；检查本人预约时排除自身。'
    add('parking','simple','simple','端点相接的车位资格','帮我查指定车位在这个时段能不能用，我需要充电。',availability,copy.deepcopy(tables),{'bay_id':'B1','start':660,'end':750,'charger':True},parking_price)
    add('parking','ordinary','ordinary','车位与区间占用关联检索','查全这段时间能用的充电车位，给我车位编号。',availability,copy.deepcopy(tables),{'start':630,'end':690,'charger':True},parking_price)
    booking_policy=availability+'按用户指定预约顺序处理，必须先检查每条指定预约的车位资格和钱包余额，向用户独立确认预约编号、时段与费用后再办；新预约办理只允许 status=pending；续时办理只允许 status=booked。本次任务对每个指定预约最多办理一次。原子把该预约设 booked、paid_cents 加本次费用并扣钱包。不可用或拒绝保持不变，前一笔成功不回滚。计费参数固定为首小时300分、后续半小时120分、封顶1500分。'
    tbl=copy.deepcopy(tables)+[table('wallets',[{'id':'W1','cents':1000}])]
    add('parking','reserve','complex','逐项停车预约和钱包事务','这两条预约按顺序帮我办，需要充电，每条先确认再付钱；办不了的不要影响办好的，最后告诉我办成的预约编号和钱包余额。',booking_policy,tbl,{'booking_ids':['N1','N2'],'wallet_id':'W1','charger':True,'approve_0':True,'approve_1':True},parking_price)
    ext=copy.deepcopy(tbl);ext[1]['rows'][1].update(status='booked',paid_cents=420);ext[1]['rows'][2].update(status='booked',paid_cents=420)
    ext[1]['rows'].append({'id':'E2','bay_id':'B2','start':760,'end':800,'status':'booked','paid_cents':300})
    extension=booking_policy+'续时任务把每条预约 end 加用户给的分钟数，原 start 不变；按新总时长费用减旧总时长费用收差额，独立确认后同时修改 end、paid_cents 和钱包。仅端点相接仍允许。'
    add('parking','extend','complex','续时冲突与差额原子扣款','两条现有预约都想延长半小时，不需要充电；分别告诉我差额并确认，冲突的别改，最后给我办成的编号和余额。',extension,ext,{'booking_ids':['N1','N2'],'extension_minutes':30,'wallet_id':'W1','charger':False,'approve_0':True,'approve_1':True},parking_price)

    returns=[{'units':4,'credit_cents':70,'quality':8},{'units':3,'credit_cents':120,'quality':4}]
    add('warehouse','direct','direct','按质退料抵扣与处理费','这批退料按这些品质和单价，扣完处理费能抵多少分？','只有 quality 不低于 minimum_quality 的退料计入 units*credit_cents，合计后扣 handling_cents，最低为0；费用单位分。',[],{},dict(returns=returns,minimum_quality=6,handling_cents=35),dict(returns=returns,minimum_quality=6,handling_cents=35))
    lots=[{'id':'L1','sku':'X','expiry_day':40,'available':4,'quarantined':False,'recalled':False},
          {'id':'L2','sku':'X','expiry_day':42,'available':8,'quarantined':True,'recalled':False},
          {'id':'L3','sku':'X','expiry_day':41,'available':7,'quarantined':False,'recalled':False},
          {'id':'L4','sku':'X','expiry_day':44,'available':9,'quarantined':False,'recalled':False},
          {'id':'L5','sku':'X','expiry_day':39,'available':99,'quarantined':False,'recalled':False}]
    lot_policy='当前业务日固定为第40日；当天到期仍可用。仅 SKU 匹配、available>0、未 quarantined、未 recalled 且 expiry_day>=40 的批次可用。分配按 expiry_day 升序，同日按批次ID字典序，从每批取不超过可用数直到满足数量。'
    add('warehouse','simple','simple','隔离批次可用性','查一下我指定的这批物料现在能不能用。',lot_policy,[table('lots',copy.deepcopy(lots))],{'lot_id':'L2','sku':'X'},{'day':40})
    add('warehouse','ordinary','ordinary','先到期先出分配与缺口','帮我查全可用库存，按先到期先出分配这次需要的数量，给我每批数量和缺口，先别扣库存。',lot_policy,[table('lots',copy.deepcopy(lots))],{'sku':'X','units':25},{'day':40})
    stock_write=lot_policy+'先完整检索可用批次，计算上述分配。只允许尚未 reserved 的订单新办预留；已 reserved 的订单不得再次扣库存，保持状态并返回空的新分配和已有预留数量。库存不足不得部分占用，返回空分配及该订单已预留数量0；库存足够时向用户确认订单和完整分配，确认后一次原子扣减所有选中批次 available，orders.reserved=true、units=请求数量，并把 capacity.reserved_units 加请求数量。拒绝则所有状态不变。'
    extra=[table('orders',[{'id':'O1','reserved':False,'units':0}]),table('capacity',[{'id':'C1','reserved_units':0}])]
    add('warehouse','reserve','complex','跨批次原子预留','帮这个订单预留所需物料，按先到期先出，给我完整分配确认后再扣库存；不够就一批都别扣，最后告诉我分配和订单预留数量。',stock_write,[table('lots',copy.deepcopy(lots)),*copy.deepcopy(extra)],{'sku':'X','units':15,'order_id':'O1','capacity_id':'C1','approve_0':True},{'day':40})
    large=generated('lots',384,{'id':seq(0,prefix='L',width=4),'sku':constant('X'),'expiry_day':cycle([39,40,41,42,43]),'available':constant(3),'quarantined':cycle([False,False,True]),'recalled':cycle([False,False,False,True])})
    add('warehouse','large','complex','大规模批次筛选与原子预留','查全这些批次，按先到期先出给订单预留18件，完整分配给我确认后一次办好，告诉我分配和已预留数量。',stock_write,[large,*copy.deepcopy(extra)],{'sku':'X','units':18,'order_id':'O1','capacity_id':'C1','approve_0':True},{'day':40})

    versions=[{'blob_id':'H1','bytes':100,'age_days':5,'pinned':False},{'blob_id':'H1','bytes':100,'age_days':8,'pinned':False},{'blob_id':'H2','bytes':200,'age_days':45,'pinned':True},{'blob_id':'H3','bytes':300,'age_days':31,'pinned':False}]
    add('archive','direct','direct','保留版本的物理空间去重','这些版本按保留策略留下后，实际占多少字节？','age_days<=retain_days 或 pinned=true 的版本保留；相同 blob_id 指向相同物理内容，字节只计一次。同 blob_id 的 bytes 一致。',[],{},dict(versions=versions,retain_days=30),dict(versions=versions,retain_days=30))
    v=[{'id':f'V{i:02d}','parent_id':f'V{i-1:02d}' if i else '', 'blob_id':f'H{i:02d}','age_days':50+i,'pinned':False,'live':True} for i in range(12)]
    b=[{'id':f'H{i:02d}','bytes':100+i,'stored':True} for i in range(12)]
    archive_policy='保留期30天：age_days<=30 或 pinned=true 的版本受保护。版本通过 parent_id 指向直接父版本，空串表示无父；链无环。删除只将 live 设 false，不删除记录。仍 live 的子版本会阻止父版本被删。字节按物理 blobs 表计数，共享 blob_id 只计一次。'
    add('archive','simple','simple','被后续版本引用的删除资格','帮我查指定版本现在能不能单独删除。',archive_policy,[table('versions',copy.deepcopy(v))],{'version_id':'V05'},{'retain_days':30})
    ordinary_v=copy.deepcopy(v);ordinary_v[5]['blob_id']='H04'
    add('archive','ordinary','ordinary','版本链追溯与共享物理内容计量','从指定版本查到最初版本，给我完整顺序链，并算出这些版本去重后的物理字节总量。',archive_policy,[table('versions',ordinary_v),table('blobs',copy.deepcopy(b))],{'version_id':'V11'},{'retain_days':30})
    archive_write=archive_policy+'清理候选集合后求最大可删子集：先去掉受保护版本；只要集合外仍 live 的版本引用集合内某个父版本，就把该父版本也移出，重复直到稳定。用户确认完整可删清单后一次原子把这些版本设非live，仅在没有任何live版本引用某blob时把它 stored=false，并把 quotas.used_bytes 改成所有仍stored的blob字节和。物理回收范围包含本工作区所有无live版本引用的blob，包括候选集合外原已无引用的blob；只有可删集合非空且用户确认后才执行整个原子操作，否则所有状态保持。拒绝全部保持。'
    tbl=[table('versions',copy.deepcopy(v)),table('blobs',copy.deepcopy(b)),table('quotas',[{'id':'Q1','used_bytes':sum(x['bytes'] for x in b)}])]
    add('archive','chain','complex','版本链清理与物理配额事务','查清这条完整版本链，把能清理的清单给我确认后再清理，别删还被保留版本依赖的内容，最后告诉我删了哪些和实际占用字节。',archive_write+'本任务候选集合为指定版本及其全部祖先。',tbl,{'version_id':'V11','quota_id':'Q1','approve_0':True},{'retain_days':30})
    large_v=[{'id':f'V{i:04d}','parent_id':'','blob_id':f'H{i//2:04d}','age_days':i%60,'pinned':i%17==0,'live':True} for i in range(384)]
    large_b=[{'id':f'H{i:04d}','bytes':10+i,'stored':True} for i in range(192)]
    # Retained live versions include shared physical content, so deleting an
    # old version does not necessarily release its blob or quota.
    for row in large_v:
        if int(row['id'][1:])>=24: row['pinned']=True
    tbl=[table('versions',large_v),table('blobs',large_b),table('quotas',[{'id':'Q1','used_bytes':sum(x['bytes'] for x in large_b)}])]
    for row in large_v[:24]: row['age_days']=50
    add('archive','large','complex','大规模版本与共享物理内容回收','查全版本和共享内容，把超过保留期且能安全清理的清单给我确认，办好后告诉我删了哪些和剩余物理占用。',archive_write+'本任务候选集合为全部live且不受保护的版本；必须查全版本记录和共享内容关系后决定。',tbl,{'quota_id':'Q1','approve_0':True},{'retain_days':30})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from prepare_v2_natural_pilot import cases as original
    from prepare_v2_cross_business import cases as cross
    pack={'cases':cases(),'historical_business_rules':[{'title':r['title'],'policy':r['business']['policy']} for r in original()+cross()]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(pack,ensure_ascii=False,indent=2)+'\n')
